import Foundation

// ---------------------------------------
// classification
// ---------------------------------------
public class Node: CustomStringConvertible {
    public var model: Model
    public var depth: Int

    // split point
    public var splitIndex: Int? = nil          // feature index
    public var splitValue: Double? = nil       // split point

    // leaf
    public var probabilities: [Double]? = nil  // probability of each output class
    public var outputIndex: Int?               // index of the highest probability output

    // node
    public var left: Node?  = nil              // < splitValue
    public var right: Node? = nil              // >= splitValue

    init(model: Model, depth: Int) {
        self.model = model
        self.depth = depth
    }
    
    public var leaf: Bool {
        return left == nil && right == nil
    }

    /// String description of the node is a weka-style format
    public var description: String {
        guard let left = self.left, right = self.right else {
            fatalError("Cannot generate description on node with no children")
        }

        guard let splitIndex = self.splitIndex, _ = self.splitValue else {
            fatalError("Cannot generate description on untrained node")
        }

        // weka style description
        var linePrefix = Array<String>(count: depth, repeatedValue: "|   ").joinWithSeparator("")
        linePrefix += model.features[splitIndex]

        return  describeChild(left, prefix: linePrefix, comparator: "<") +
                describeChild(right, prefix: linePrefix, comparator: ">=")
    }

    internal func describeChild(node: Node, prefix: String, comparator: String) -> String {
        let prefix = "\(prefix) \(comparator) \(splitValue!)"

        if node.leaf {
            guard let index = node.outputIndex, probabilities = node.probabilities else {
                fatalError("Cannot generate description for leaf with no output")
            }

            return "\(prefix) : \(model.outputs[index]) \(probabilities)\n"

        } else {
            return "\(prefix)\n\(node.description)"
        }
    }
    
    public func train() {
        fatalError("Only trainable nodes can be trained")
    }
}



// ---------------------------------------
// training
// ---------------------------------------
/// Split represents a potential split point on a feature
final internal class Split {
    // split value
    let featureIndex: Int
    let splitValue: Double
    
    // split examples
    let left:  SubSet
    let right: SubSet

    // cached entropy score
    var score = 0.0
    
    init(featureIndex: Int, range: (min: Double, max: Double), tree: TrainableTree) {
        self.featureIndex = featureIndex

        // randomly pick a split point between min..max
        let interval = range.max - range.min
        self.splitValue = (erand48(&tree.xsubi) * interval) + range.min
        
        // examples will be stores in left (< value) or right (>= value) subsets
        self.left  = SubSet(model: tree.model)
        self.right = SubSet(model: tree.model)
    }
    
    func addExample(example: Example) {
        if example.values[featureIndex] < splitValue {
            left.append(example)
        } else {
            right.append(example)
        }
    }
    
    func entropy() -> Double {
        let totalCount = Double(left.examples.count + right.examples.count)
        let Hl = left.entropy() * (Double(left.examples.count) / totalCount)
        let Hr = right.entropy() * (Double(right.examples.count) / totalCount)
        return Hl + Hr
    }

    // weighted information gain as outlined in `Extremely randomized trees'
    func calculateInformationGain(currentEntropy: Double) {
        let Hc = currentEntropy
        let Hs = entropy() // split entropy
        score = (2 * (Hc - Hs)) / (Hc + Hs)
    }
}


/// Trainable nodes are valid nodes, but implement split to support model creation
final public class TrainableNode: Node {
    public var tree: TrainableTree
    public var subset: SubSet

    public init(tree: TrainableTree, depth: Int, subset: SubSet) {
        self.subset = subset
        self.tree = tree
        super.init(model: tree.model, depth: depth)
    }

    /// Train this node on the examples in `subset'. The node will either become a
    /// leaf node with output probabilities, or an internal node with children
    override public func train() {
        if shouldProduceLeaf() {
            calculateOutputProbabilities()
        } else {
            split()
        }
    }
    
    /// Determines whether the current set of examples in subset warrant creating a
    /// leaf node. When the node cannot be split further (because it has too few
    /// examples, or splitting won't create any new valid paths because all output
    /// classes are the same or all example values are the same) a leaf is created.
    func shouldProduceLeaf() -> Bool {
        return  subset.count < tree.minExamples ||
                subset.allEqualOutputs ||
                subset.allEqualValues
    }
    
    /// Calculate the output class probabilities, and keep track of the output with
    /// the highest probability for classification.
    func calculateOutputProbabilities() {
        // subset counts instances of each output class
        var probabilities = subset.outputCounts.map { Double($0) }
        let examplesCount = Double(subset.examples.count)
        
        // divide through by the number of examples to produce probabilities and
        // keep track of the most probable output class
        var maxProb = 0.0
        var maxIndex = 0

        for i in 0..<probabilities.count {
            let prob = probabilities[i] / examplesCount
            probabilities[i] = prob
            
            // keep track of highest probability output class
            if prob > maxProb {
                maxProb = prob
                maxIndex = i
            }
        }
        
        self.probabilities = probabilities
        self.outputIndex = maxIndex

        // depth and breadth tracking
        tree.maxDepth = max(tree.maxDepth, depth)
        tree.numLeaves += 1
    }
    
    /// Randomly select K features from the model and produce a random split point
    /// on each feature. Information gain is used to score each potential split and
    /// the best performing split is selected. The examples in `subset' are split
    /// between new left and right child nodes according to the split value, and
    /// training recursively continues down the tree.
    func split() {
        var ranges = subset.featureRanges()
        var indexes = [Int](0..<ranges.count)
        
        // randomly select features to score
        for _ in 0..<(ranges.count - tree.pickFeatures) {
            let index = nrand48(&tree.xsubi) % indexes.count
            indexes.removeAtIndex(index)
            ranges.removeAtIndex(index)
        }
        
        // create random splits for each selected feature
        var splits = [Split]()
        for (index, range) in zip(indexes, ranges) {
            splits.append(
                Split(featureIndex: index, range: range, tree: tree)
            )
        }
        
        // collect subsets for each potential split
        for example in subset.examples {
            for split in splits {
                split.addExample(example)
            }
        }
        
        // score each split using weighted information gain
        let Hc = subset.entropy()
        for split in splits {
            split.calculateInformationGain(Hc)
        }
        
        // pick the highest scoring split and construct children
        guard let maxSplit = splits.maxElement({ $0.score < $1.score }) else { fatalError() }

        // release memory before constructing more of the tree
        splits.removeAll()

        // prepare to train on left and right subsets
        splitIndex = maxSplit.featureIndex
        splitValue = maxSplit.splitValue
        left  = TrainableNode(tree: tree, depth: depth + 1, subset: maxSplit.left)
        right = TrainableNode(tree: tree, depth: depth + 1, subset: maxSplit.right)
        
        // recursively continue constructing the tree
        left!.train()
        right!.train()
    }
}
