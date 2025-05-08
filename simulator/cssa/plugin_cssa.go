package cssa

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"golang.org/x/xerrors"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

// CSSA is a scheduler plugin that implements the Chaotic Squirrel Search Algorithm
type CSSA struct {
	handle framework.Handle
	// CSSA parameters
	populationSize int
	maxIterations  int
	pMin           float64 // Minimum probability for winter phase
	jw             float64 // Initial jumping probability
	wa             float64 // Initial controlled jumps probability
	yd             float64 // Initial exploration rate
	fFactor        float64 // Decreasing factor
	eps            float64 // Small constant to avoid division by zero
}

// KubernetesTask represents a pod to be scheduled
type KubernetesTask struct {
	name          string
	namespace     string
	cpuRequest    float64
	memoryRequest float64
	gpuRequest    int
	priorityClass string
	labels        map[string]string
	nodeName      string
	priorityScore float64
	pod           *v1.Pod // Reference to the original pod
}

// KubernetesNode represents a node in the cluster
type KubernetesNode struct {
	name             string
	allocatableCPU   float64
	allocatableMemory float64
	allocatableGPU   int
	labels           map[string]string
	taints           []v1.Taint
	scheduledPods    []*KubernetesTask
	availableCPU     float64
	availableMemory  float64
	availableGPU     int
	node             *v1.Node // Reference to the original node
}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "CSSA"
	preScoreStateKey = "PreScore" + Name
)

var (
	_ framework.ScorePlugin    = &CSSA{}
	_ framework.PreScorePlugin = &CSSA{}
	_ framework.FilterPlugin   = &CSSA{}
	_ framework.PreFilterPlugin = &CSSA{}
)

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	tasks   []*KubernetesTask
	k8sNodes []*KubernetesNode
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() framework.StateData {
	return s
}

// Name returns the name of the plugin. It is used in logs, etc.
func (c *CSSA) Name() string {
	return Name
}

func (c *CSSA) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	klog.InfoS("execute PreFilter on CSSA plugin", "pod", klog.KObj(pod))
	
	// Convert pod to internal representation
	tasks := c.convertPodsToTasks([]*v1.Pod{pod})
	
	if len(tasks) == 0 {
		return nil, framework.NewStatus(framework.Error, "failed to convert pod to task")
	}

	s := &preScoreState{
		tasks: tasks,
	}
	state.Write(preScoreStateKey, s)
	
	return nil, nil
}

func (c *CSSA) PreFilterExtensions() framework.PreFilterExtensions {
    return nil
}

// PreScore computes pod and node information needed for scoring phase
func (c *CSSA) PreScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodes []*framework.NodeInfo) *framework.Status {
	klog.InfoS("execute PreScore on CSSA plugin", "pod", klog.KObj(pod))
	
	// Convert pods and nodes to our internal representation
	tasks := c.convertPodsToTasks([]*v1.Pod{pod})
	k8sNodes := c.convertNodesToKubernetesNodes(nodes)
	
	s := &preScoreState{
		tasks:   tasks,
		k8sNodes: k8sNodes,
	}
	state.Write(preScoreStateKey, s)
	
	return nil
}

// EventsToRegister returns the events needed to be registered 
func (c *CSSA) EventsToRegister() []framework.ClusterEvent {
	return []framework.ClusterEvent{
		{Resource: framework.Node, ActionType: framework.Add},
		{Resource: framework.Pod, ActionType: framework.Add},
		{Resource: framework.Pod, ActionType: framework.Update},
	}
}

// Score invokes the CSSA algorithm and returns a score for the node
func (c *CSSA) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	klog.InfoS("execute Score on CSSA plugin", "pod", klog.KObj(pod))
	
	data, err := state.Read(preScoreStateKey)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("failed to read preScoreState: %v", err))
	}

	s, ok := data.(*preScoreState)
	if !ok {
		err = xerrors.Errorf("fetched pre score state is not *preScoreState, but %T", data)
		return 0, framework.AsStatus(err)
	}
	
	tasks := s.tasks
	k8sNodes := s.k8sNodes

	// Find the node we're scoring
	var scoringNodeIdx int = -1
	for i, node := range k8sNodes {
		if node.name == nodeName {
			scoringNodeIdx = i
			break
		}
	}

	if scoringNodeIdx == -1 {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("node %s not found", nodeName))
	}

	// Run CSSA optimization
	solution, _ := c.optimize(tasks, k8sNodes)

	// Calculate score based on whether this node was chosen for the pod
	score := int64(0)
	for i, nodeIdx := range solution {
		if nodeIdx == scoringNodeIdx && i < len(tasks) {
			// This node was chosen for a pod, give it a high score
			score = framework.MaxNodeScore
			break
		}
	}

	return score, nil
}

// ScoreExtensions returns the score extension interface
func (c *CSSA) ScoreExtensions() framework.ScoreExtensions {
	return c
}

// NormalizeScore normalizes the scores to the range [0, framework.MaxNodeScore]
func (c *CSSA) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	// Find the highest and lowest scores
	var highest int64 = 0
	var lowest int64 = framework.MaxNodeScore

	for _, nodeScore := range scores {
		if nodeScore.Score > highest {
			highest = nodeScore.Score
		}
		if nodeScore.Score < lowest {
			lowest = nodeScore.Score
		}
	}

	// If all scores are the same, return as is
	if highest == lowest {
		return nil
	}

	// Normalize the scores
	for i := range scores {
		scores[i].Score = (scores[i].Score * framework.MaxNodeScore) / highest
	}

	return nil
}

// Filter checks if a node can accommodate the pod
func (c *CSSA) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
    node := nodeInfo.Node()
    if node == nil {
        return framework.NewStatus(framework.Error, "node not found")
    }
    
    // Convert pod and node to internal representation
    taskPod := c.convertPodsToTasks([]*v1.Pod{pod})[0]
    k8sNode := &KubernetesNode{
        name:             node.Name,
        allocatableCPU:   float64(node.Status.Allocatable.Cpu().MilliValue()) / 1000.0,
        allocatableMemory: float64(node.Status.Allocatable.Memory().Value()) / (1024 * 1024),
        allocatableGPU:   int(node.Status.Allocatable.Name("nvidia.com/gpu", "0").Value()),
        labels:           node.Labels,
        taints:           node.Spec.Taints,
        availableCPU:     float64(node.Status.Allocatable.Cpu().MilliValue()) / 1000.0,
        availableMemory:  float64(node.Status.Allocatable.Memory().Value()) / (1024 * 1024),
        availableGPU:     int(node.Status.Allocatable.Name("nvidia.com/gpu", "0").Value()),
        node:             node,
    }
    
    // Check if node can accommodate pod
    if !c.canNodeAccommodatePod(k8sNode, taskPod) {
        return framework.NewStatus(framework.Unschedulable, "node cannot accommodate pod")
    }
    
    return framework.NewStatus(framework.Success, "")
}

// CSSAArgs are the arguments for the CSSA plugin
type CSSAArgs struct {
	metav1.TypeMeta

	// PopulationSize defines the size of the population for CSSA algorithm
	PopulationSize int `json:"populationSize,omitempty"`
	
	// MaxIterations defines the maximum number of iterations for CSSA algorithm
	MaxIterations int `json:"maxIterations,omitempty"`
}

// New initializes a new plugin and returns it
func New(ctx context.Context, arg runtime.Object, h framework.Handle) (framework.Plugin, error) {
	typedArg := CSSAArgs{
		PopulationSize: 30,
		MaxIterations: 100,
	}
	
	if arg != nil {
		err := frameworkruntime.DecodeInto(arg, &typedArg)
		if err != nil {
			return nil, xerrors.Errorf("decode arg into CSSAArgs: %w", err)
		}
		klog.InfoS("CSSAArgs is successfully applied", 
			"populationSize", typedArg.PopulationSize, 
			"maxIterations", typedArg.MaxIterations)
	}
	
	return &CSSA{
		handle:         h,
		populationSize: typedArg.PopulationSize,
		maxIterations:  typedArg.MaxIterations,
		pMin:           0.15, // Minimum probability for winter phase
		jw:             0.4,  // Initial jumping probability
		wa:             0.8,  // Initial controlled jumps probability
		yd:             0.6,  // Initial exploration rate
		fFactor:        0.9,  // Decreasing factor
		eps:            0.001, // Small constant to avoid division by zero
	}, nil
}
// convertPodsToTasks converts Kubernetes pods to KubernetesTasks
func (c *CSSA) convertPodsToTasks(pods []*v1.Pod) []*KubernetesTask {
	tasks := make([]*KubernetesTask, 0, len(pods))

	for _, pod := range pods {
		cpuRequest := c.getResourceRequest(pod, v1.ResourceCPU)
		memoryRequest := c.getResourceRequest(pod, v1.ResourceMemory)
		gpuRequest := int(c.getResourceRequest(pod, "nvidia.com/gpu"))

		task := &KubernetesTask{
			name:          pod.Name,
			namespace:     pod.Namespace,
			cpuRequest:    cpuRequest,
			memoryRequest: memoryRequest,
			gpuRequest:    gpuRequest,
			priorityClass: getPriorityClassName(pod),
			labels:        pod.Labels,
			pod:           pod,
		}

		// Calculate priority score
		cpuScore := 1.0 / (task.cpuRequest + 1)
		memoryScore := 1.0 / (task.memoryRequest + 1)
		gpuScore := 1.0
		if task.gpuRequest > 0 {
			gpuScore = 1.0 / float64(task.gpuRequest+1)
		}

		task.priorityScore = 0.4*cpuScore + 0.4*memoryScore + 0.2*gpuScore
		tasks = append(tasks, task)
	}

	// Sort tasks by priority score in descending order
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].priorityScore > tasks[j].priorityScore
	})

	return tasks
}

// getResourceRequest returns the resource request for a pod
func (c *CSSA) getResourceRequest(pod *v1.Pod, resourceName v1.ResourceName) float64 {
	total := 0.0
	for _, container := range pod.Spec.Containers {
		if container.Resources.Requests != nil {
			if val, ok := container.Resources.Requests[resourceName]; ok {
				total += float64(val.MilliValue()) / 1000.0
			}
		}
	}
	return total
}

// getPriorityClassName returns the priority class name for a pod
func getPriorityClassName(pod *v1.Pod) string {
	if pod.Spec.PriorityClassName != "" {
		return pod.Spec.PriorityClassName
	}
	return ""
}

// convertNodesToKubernetesNodes converts Kubernetes nodes to KubernetesNodes
func (c *CSSA) convertNodesToKubernetesNodes(nodeInfos []*framework.NodeInfo) []*KubernetesNode {
	nodes := make([]*KubernetesNode, 0, len(nodeInfos))

	for _, nodeInfo := range nodeInfos {
		node := nodeInfo.Node()
		if node == nil {
			continue
		}

		// Skip unschedulable nodes
		if node.Spec.Unschedulable {
			continue
		}

		allocatableCPU := float64(node.Status.Allocatable.Cpu().MilliValue()) / 1000.0
		allocatableMemory := float64(node.Status.Allocatable.Memory().Value()) / (1024 * 1024) // Convert to MB
		allocatableGPU := int(node.Status.Allocatable.Name("nvidia.com/gpu", "0").Value())

		k8sNode := &KubernetesNode{
			name:             node.Name,
			allocatableCPU:   allocatableCPU,
			allocatableMemory: allocatableMemory,
			allocatableGPU:   allocatableGPU,
			labels:           node.Labels,
			taints:           node.Spec.Taints,
			scheduledPods:    []*KubernetesTask{},
			availableCPU:     allocatableCPU,
			availableMemory:  allocatableMemory,
			availableGPU:     allocatableGPU,
			node:             node,
		}

		nodes = append(nodes, k8sNode)
	}

	return nodes
}

// optimize implements the CSSA optimization algorithm
func (c *CSSA) optimize(pods []*KubernetesTask, nodes []*KubernetesNode) ([]int, float64) {
	if len(pods) == 0 || len(nodes) == 0 {
		return []int{}, 0.0
	}

	// Initialize population of solutions
	population := c.initializePopulation(pods, nodes)

	bestSolution := make([]int, len(pods))
	bestFitness := math.Inf(1)

	// CSSA Algorithm main loop
	for iteration := 0; iteration < c.maxIterations; iteration++ {
		// Calculate fitness for each solution
		fitnessScores := make([]float64, len(population))
		for i, solution := range population {
			fitnessScores[i] = c.calculateFitness(solution, pods, nodes)
		}

		// Update best solution
		minFitnessIdx := 0
		for i, fitness := range fitnessScores {
			if fitness < fitnessScores[minFitnessIdx] {
				minFitnessIdx = i
			}
		}

		if fitnessScores[minFitnessIdx] < bestFitness {
			bestFitness = fitnessScores[minFitnessIdx]
			copy(bestSolution, population[minFitnessIdx])
		}

		// Update dynamic parameters based on iteration progress
		c.updateParameters(iteration)

		// CSSA phases with progressive search
		newPopulation := make([][]int, len(population))
		for i := range population {
			// Determine current phase based on progressive strategy
			if rand.Float64() < c.pMin {
				// Winter phase (exploitation): Local search
				newPopulation[i] = c.progressiveWinterPhase(population[i], fitnessScores[i], iteration, pods, nodes)
			} else {
				// Summer phase (exploration): Global search
				newPopulation[i] = c.progressiveSummerPhase(population[i], population, fitnessScores, iteration, pods, nodes)
			}

			// Apply jumping behavior with dynamic control
			if rand.Float64() < c.jw {
				newPopulation[i] = c.jumpingSearch(newPopulation[i], iteration, pods, nodes)
			}

			// Ensure solution feasibility
			newPopulation[i] = c.repairSolution(newPopulation[i], pods, nodes)
		}

		population = newPopulation
	}

	return bestSolution, bestFitness
}

// initializePopulation initializes a population of valid scheduling solutions
func (c *CSSA) initializePopulation(pods []*KubernetesTask, nodes []*KubernetesNode) [][]int {
	population := make([][]int, c.populationSize)

	for p := 0; p < c.populationSize; p++ {
		solution := make([]int, len(pods))
		nodesCopy := c.copyNodes(nodes)

		for i, pod := range pods {
			// Find suitable nodes for this pod
			suitableNodes := []int{}
			for j, node := range nodesCopy {
				if c.canNodeAccommodatePod(node, pod) {
					suitableNodes = append(suitableNodes, j)
				}
			}

			if len(suitableNodes) > 0 {
				// Randomly select a suitable node
				nodeIdx := suitableNodes[rand.Intn(len(suitableNodes))]
				solution[i] = nodeIdx

				// Update node resources
				nodesCopy[nodeIdx].availableCPU -= pod.cpuRequest
				nodesCopy[nodeIdx].availableMemory -= pod.memoryRequest
				nodesCopy[nodeIdx].availableGPU -= pod.gpuRequest
			} else {
				// No suitable node found
				solution[i] = -1
			}
		}

		population[p] = solution
	}

	return population
}

// copyNodes creates a deep copy of nodes
func (c *CSSA) copyNodes(nodes []*KubernetesNode) []*KubernetesNode {
	nodesCopy := make([]*KubernetesNode, len(nodes))
	for i, node := range nodes {
		nodesCopy[i] = &KubernetesNode{
			name:             node.name,
			allocatableCPU:   node.allocatableCPU,
			allocatableMemory: node.allocatableMemory,
			allocatableGPU:   node.allocatableGPU,
			labels:           node.labels,
			taints:           node.taints,
			scheduledPods:    []*KubernetesTask{},
			availableCPU:     node.allocatableCPU,
			availableMemory:  node.allocatableMemory,
			availableGPU:     node.allocatableGPU,
			node:             node.node,
		}
	}
	return nodesCopy
}

// canNodeAccommodatePod checks if a node can accommodate a pod
func (c *CSSA) canNodeAccommodatePod(node *KubernetesNode, pod *KubernetesTask) bool {
	// Check resources
	if node.availableCPU < pod.cpuRequest ||
		node.availableMemory < pod.memoryRequest ||
		node.availableGPU < pod.gpuRequest {
		return false
	}

	// Check node affinity (simplified)
	if nodeAffinity, ok := pod.labels["kubernetes.io/node-affinity"]; ok {
		requiredLabels := parseNodeAffinity(nodeAffinity)
		for key, value := range requiredLabels {
			if node.labels[key] != value {
				return false
			}
		}
	}

	// Check taints and tolerations (simplified)
	for _, taint := range node.taints {
		if !podToleratesTaint(pod, taint) {
			return false
		}
	}

	return true
}

// parseNodeAffinity is a helper function to parse node affinity string
func parseNodeAffinity(affinityStr string) map[string]string {
	// This is a simplified implementation of parsing affinity string
	// In a real implementation, you would parse the string based on your format
	return map[string]string{}
}

// podToleratesTaint checks if a pod tolerates a taint
func podToleratesTaint(pod *KubernetesTask, taint v1.Taint) bool {
	// This is a simplified implementation
	// In a real implementation, you would check the pod's tolerations
	return false
}

// calculateFitness calculates fitness as a combination of makespan and resource utilization
func (c *CSSA) calculateFitness(solution []int, pods []*KubernetesTask, nodes []*KubernetesNode) float64 {
	// Check for invalid solution with unscheduled pods
	for _, nodeIdx := range solution {
		if nodeIdx == -1 {
			return math.Inf(1)
		}
	}

	nodesCopy := c.copyNodes(nodes)

	// Track load on each node
	cpuLoads := make([]float64, len(nodesCopy))
	memoryLoads := make([]float64, len(nodesCopy))
	gpuLoads := make([]float64, len(nodesCopy))

	// Calculate resource utilization
	for i, nodeIdx := range solution {
		if nodeIdx != -1 && i < len(pods) { // Pod is scheduled
			pod := pods[i]

			cpuLoads[nodeIdx] += pod.cpuRequest
			memoryLoads[nodeIdx] += pod.memoryRequest
			gpuLoads[nodeIdx] += float64(pod.gpuRequest)
		}
	}

	// Calculate normalized makespan (maximum load)
	normalizedCPULoads := make([]float64, len(nodesCopy))
	normalizedMemoryLoads := make([]float64, len(nodesCopy))
	normalizedGPULoads := make([]float64, len(nodesCopy))

	for i, node := range nodesCopy {
		normalizedCPULoads[i] = cpuLoads[i] / node.allocatableCPU
		normalizedMemoryLoads[i] = memoryLoads[i] / node.allocatableMemory
		if node.allocatableGPU > 0 {
			normalizedGPULoads[i] = gpuLoads[i] / float64(node.allocatableGPU)
		} else {
			normalizedGPULoads[i] = 0
		}
	}

	// Makespan is the maximum normalized load
	cpuMakespan := c.max(normalizedCPULoads)
	memoryMakespan := c.max(normalizedMemoryLoads)
	gpuMakespan := c.max(normalizedGPULoads)

	// Calculate load imbalance (standard deviation)
	cpuImbalance := c.standardDeviation(normalizedCPULoads)
	memoryImbalance := c.standardDeviation(normalizedMemoryLoads)
	gpuImbalance := c.standardDeviation(normalizedGPULoads)

	// Combined fitness value (lower is better)
	makespan := 0.4*cpuMakespan + 0.4*memoryMakespan + 0.2*gpuMakespan
	imbalance := 0.4*cpuImbalance + 0.4*memoryImbalance + 0.2*gpuImbalance

	return 0.7*makespan + 0.3*imbalance
}

// max returns the maximum value in a slice
func (c *CSSA) max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

// standardDeviation calculates the standard deviation of a slice of values
func (c *CSSA) standardDeviation(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate variance
	variance := 0.0
	for _, v := range values {
		variance += math.Pow(v-mean, 2)
	}
	variance /= float64(len(values))

	// Return standard deviation
	return math.Sqrt(variance)
}

// updateParameters updates algorithm parameters based on iteration progress
func (c *CSSA) updateParameters(iteration int) {
	// Calculate progress ratio
	progress := float64(iteration) / float64(c.maxIterations)

	// Update parameters based on pseudocode
	c.pMin = math.Max(0.05, c.pMin*c.fFactor)        // Decreasing winter phase probability
	c.jw = math.Min(0.9, c.jw/c.fFactor)             // Increasing jumping probability
	c.yd = math.Max(0.1, c.yd*c.fFactor)             // Decreasing exploration rate
	c.wa = 0.9 - 0.8*progress                        // Decreasing control value
}

// progressiveWinterPhase implements winter phase with progressive search (local exploitation)
func (c *CSSA) progressiveWinterPhase(solution []int, fitness float64, iteration int, pods []*KubernetesTask, nodes []*KubernetesNode) []int {
	newSolution := make([]int, len(solution))
	copy(newSolution, solution)

	// Progressive search intensity based on iteration
	progress := float64(iteration) / float64(c.maxIterations)
	intensity := int(math.Max(1, 3*(1-progress))) // More moves early, fewer later

	for i := 0; i < intensity; i++ {
		// Select random pod to move
		podIdx := rand.Intn(len(solution))
		currentNode := solution[podIdx]

		// Find alternative nodes
		pod := pods[podIdx]
		suitableNodes := []int{}

		for i, node := range nodes {
			if i != currentNode {
				nodeCopy := &KubernetesNode{
					name:             node.name,
					allocatableCPU:   node.allocatableCPU,
					allocatableMemory: node.allocatableMemory,
					allocatableGPU:   node.allocatableGPU,
					labels:           node.labels,
					taints:           node.taints,
					availableCPU:     node.availableCPU,
					availableMemory:  node.availableMemory,
					availableGPU:     node.availableGPU,
				}
				if c.canNodeAccommodatePod(nodeCopy, pod) {
					suitableNodes = append(suitableNodes, i)
				}
			}
		}

		if len(suitableNodes) > 0 {
			// Move pod to random suitable node
			newNode := suitableNodes[rand.Intn(len(suitableNodes))]
			newSolution[podIdx] = newNode

			// Check if move improves fitness
			newFitness := c.calculateFitness(newSolution, pods, nodes)

			// Accept only if improved (exploitation)
			if newFitness >= fitness {
				newSolution[podIdx] = currentNode // Revert the move
			}
		}
	}

	return newSolution
}

// progressiveSummerPhase implements summer phase with progressive search (global exploration)
func (c *CSSA) progressiveSummerPhase(solution []int, population [][]int, fitnessScores []float64, iteration int, pods []*KubernetesTask, nodes []*KubernetesNode) []int {
	newSolution := make([]int, len(solution))
	copy(newSolution, solution)

	// Create indices sorted by fitness
	indices := make([]int, len(population))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return fitnessScores[indices[i]] < fitnessScores[indices[j]]
	})

	// Progressive selection probability (favor better solutions)
	selectionProb := make([]float64, len(population))
	sum := 0.0
	for i := range population {
		selectionProb[i] = math.Pow(1-float64(i)/float64(len(population)), 2)
		sum += selectionProb[i]
	}
	for i := range selectionProb {
		selectionProb[i] /= sum
	}

	// Select reference solution based on weighted random selection
	refIdx := c.weightedRandomSelection(selectionProb)
	refSolution := population[refIdx]

	// Adaptive crossover rate based on iteration
	progress := float64(iteration) / float64(c.maxIterations)
	crossoverRate := c.yd * (1 - 0.5*progress) // Higher early, lower later

	// Perform crossover with dynamic control
	for i := range solution {
		if rand.Float64() < crossoverRate {
			pod := pods[i]
			newNode := refSolution[i]

			// Check if new node is suitable
			if newNode != -1 {
				nodeCopy := &KubernetesNode{
					name:             nodes[newNode].name,
					allocatableCPU:   nodes[newNode].allocatableCPU,
					allocatableMemory: nodes[newNode].allocatableMemory,
					allocatableGPU:   nodes[newNode].allocatableGPU,
					labels:           nodes[newNode].labels,
					taints:           nodes[newNode].taints,
					availableCPU:     nodes[newNode].availableCPU,
					availableMemory:  nodes[newNode].availableMemory,
					availableGPU:     nodes[newNode].availableGPU,
				}
				if c.canNodeAccommodatePod(nodeCopy, pod) {
					newSolution[i] = newNode
				}
			}
		}
	}

	return newSolution
}

// weightedRandomSelection selects an index based on weights
func (c *CSSA) weightedRandomSelection(weights []float64) int {
	r := rand.Float64()
	sum := 0.0
	for i, w := range weights {
		sum += w
		if r <= sum {
			return i
		}
	}
	return len(weights) - 1
}

// jumpingSearch applies jumping behavior for diversity with dynamic control
func (c *CSSA) jumpingSearch(solution []int, iteration int, pods []*KubernetesTask, nodes []*KubernetesNode) []int {
	newSolution := make([]int, len(solution))
	copy(newSolution, solution)

	// Dynamic jumping intensity based on iteration progress
	progress := float64(iteration) / float64(c.maxIterations)
	jumpProbability := c.wa * (1 - progress) // Higher early, lower later

	// Vector dimension (number of pods to consider)
	dimensions := len(solution)

	// Select random dimensions to jump
	jumpDims := c.randomSample(dimensions, int(math.Max(1, float64(dimensions)*jumpProbability)))

	// Apply jumping to selected dimensions
	for _, i := range jumpDims {
		pod := pods[i]

		// Find all suitable nodes
		suitableNodes := []int{}
		for j, node := range nodes {
			nodeCopy := &KubernetesNode{
				name:             node.name,
				allocatableCPU:   node.allocatableCPU,
				allocatableMemory: node.allocatableMemory,
				allocatableGPU:   node.allocatableGPU,
				labels:           node.labels,
				taints:           node.taints,
				availableCPU:     node.availableCPU,
				availableMemory:  node.availableMemory,
				availableGPU:     node.availableGPU,
			}
			if c.canNodeAccommodatePod(nodeCopy, pod) {
				suitableNodes = append(suitableNodes, j)
			}
		}

		if len(suitableNodes) > 0 {
			// Apply jumping - choose a random node
			jumpStrength := rand.Float64() // Random jump strength

			if jumpStrength < 0.3 { // Small jump - stay in neighborhood
				for j, nodeIdx := range suitableNodes {
					if nodeIdx == solution[i] && len(suitableNodes) > 1 {
						// Remove current node from suitable nodes
						suitableNodes = append(suitableNodes[:j], suitableNodes[j+1:]...)
						break
					}
				}
			}

			// Make the jump to a new node
			if len(suitableNodes) > 0 {
				newSolution[i] = suitableNodes[rand.Intn(len(suitableNodes))]
			}
		}
	}

	return newSolution
}

// randomSample selects k random integers from range [0,n)
func (c *CSSA) randomSample(n, k int) []int {
	if k > n {
		k = n
	}

	// Fisher-Yates shuffle algorithm
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < k; i++ {
		j := rand.Intn(n-i) + i
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices[:k]
}

// repairSolution ensures solution feasibility
func (c *CSSA) repairSolution(solution []int, pods []*KubernetesTask, nodes []*KubernetesNode) []int {
	repaired := make([]int, len(solution))
	copy(repaired, solution)

	// Create a copy of nodes to track resource utilization
	nodesCopy := c.copyNodes(nodes)

	// Allocate resources for each pod
	for i, nodeIdx := range repaired {
		if nodeIdx >= 0 && nodeIdx < len(nodesCopy) {
			pod := pods[i]

			// Check if node can still accommodate pod
			if c.canNodeAccommodatePod(nodesCopy[nodeIdx], pod) {
				// Update node resources
				nodesCopy[nodeIdx].availableCPU -= pod.cpuRequest
				nodesCopy[nodeIdx].availableMemory -= pod.memoryRequest
				nodesCopy[nodeIdx].availableGPU -= pod.gpuRequest
			} else {
				// Find alternative node
				foundNode := false
				for j, node := range nodesCopy {
					if c.canNodeAccommodatePod(node, pod) {
						repaired[i] = j
						nodesCopy[j].availableCPU -= pod.cpuRequest
						nodesCopy[j].availableMemory -= pod.memoryRequest
						nodesCopy[j].availableGPU -= pod.gpuRequest
						foundNode = true
						break
					}
				}

				if !foundNode {
					// If no node can accommodate, mark as unschedulable
					repaired[i] = -1
				}
			}
		}
	}

	return repaired
}
