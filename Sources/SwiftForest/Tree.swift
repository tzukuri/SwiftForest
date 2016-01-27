import Foundation

// Implementation of:
// Extremely randomized trees
// http://www.montefiore.ulg.ac.be/%7Eernst/uploads/news/id63/extremely-randomized-trees.pdf

public protocol Classifier {
    func train(trainingSet: TrainingSet)
    func classify(values: [Double]) -> Int
    var delegate: ClassifierDelegate? { get set }
}

public protocol ClassifierDelegate {
    mutating func trainingWillStartWithSteps(count: Int)
    mutating func trainingDidCompleteStep()
    mutating func trainingDidFinish()
}

final public class Forest: Classifier {
    public var trees = [Tree]()
    public var delegate: ClassifierDelegate?
    public var randomSeed: Int
    
    public init(size: Int = 100, numFeatures: Int? = nil, minExamples: Int = 2, randomSeed: Int = 1, delegate: ClassifierDelegate? = nil) {
        self.randomSeed = randomSeed
        self.delegate = delegate

        for _ in 0..<size {
            self.trees.append(
                Tree(numFeatures: numFeatures, minExamples: minExamples)
            )
        }
    }
    
    public func train(trainingSet: TrainingSet) {
        delegate?.trainingWillStartWithSteps(trees.count)
        srand48(randomSeed)

        for tree in trees {
            tree.randomSeed = lrand48()
            tree.train(trainingSet)
            delegate?.trainingDidCompleteStep()
        }

        delegate?.trainingDidFinish()
    }
    
    public func classify(values: [Double]) -> Int {
        var counts = [Int: Int]()
        
        // classify values in each tree and count number of times
        // each output class is selected
        for tree in trees {
            let output = tree.classify(values)
            counts[output] = (counts[output] ?? 0) + 1
        }
        
        // sort output classes by count ($0 and $1 are (key, value))
        // maxElement expects result of: $0 ordered before $1
        if let mode = counts.maxElement({ $0.1 < $1.1 }) {
            return mode.0
        } else {
            return counts.values.first!
        }
    }
}

final public class Tree: Classifier {
    public var delegate: ClassifierDelegate?
    public var trainingSet: TrainingSet!
    public var numFeatures: Int!
    public var minExamples: Int
    public var randomSeed: Int
    public var root: Node!
    
    public init(numFeatures: Int? = nil, minExamples: Int = 2, randomSeed: Int = 1) {
        self.numFeatures = numFeatures
        self.minExamples = minExamples
        self.randomSeed = randomSeed
    }
    
    public func train(trainingSet: TrainingSet) {
        delegate?.trainingWillStartWithSteps(1)
        self.trainingSet = trainingSet
        let allExamples = SubSet(examples: trainingSet.examples, trainingSet: trainingSet)

        // default numFeatures to sqrt(|features|)
        if numFeatures == nil {
            numFeatures = Int(sqrt(Double(trainingSet.features.count)))
        }

        // reset the root node on train so the tree can be re-used
        root = Node()
        root.split(allExamples, tree: self)

        // there's only a single tree to build, so progress is 0.0 -> 1.0
        delegate?.trainingDidCompleteStep()
        delegate?.trainingDidFinish()
    }
    
    public func classify(values: [Double]) -> Int {
        var node: Node = root
        
        // walk the tree until we reach a leaf
        while !node.leaf {
            guard let index = node.splitIndex, value = node.splitValue else { fatalError() }
            if values[index] < value {
                node = node.left!
            } else {
                node = node.right!
            }
        }
        
        guard let index = node.outputIndex else { fatalError() }
        return index
    }
}

final internal class Split {
    let index: Int
    let value: Double
    var score = 0.0
    let left: SubSet
    let right: SubSet
    
    init(index: Int, range: (min: Double, max: Double), trainingSet: TrainingSet) {
        let interval = range.max - range.min
        self.value = (drand48() * interval) + range.min
        self.index = index
        self.left = SubSet(trainingSet: trainingSet)
        self.right = SubSet(trainingSet: trainingSet)
    }
    
    func addExample(example: Example) {
        if example.values[index] < value {
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
}

final public class Node {
    public var splitIndex: Int? = nil          // feature index
    public var splitValue: Double? = nil       // split point
    public var probabilities: [Double]? = nil  // probability of each output class
    public var outputIndex: Int?               // index of the highest probability output
    public var left: Node?  = nil              // < split
    public var right: Node? = nil              // >= split
    
    public var leaf: Bool {
        return left == nil && right == nil
    }
    
    public func split(subset: SubSet, tree: Tree) {
        if shouldProduceLeaf(subset, tree: tree) {
            calculateOutputProbabilities(subset, tree: tree)
        } else {
            randomlySplit(subset, tree: tree)
        }
    }
    
    func shouldProduceLeaf(subset: SubSet, tree: Tree) -> Bool {
        return  subset.count < tree.minExamples ||
                subset.allEqualOutputs ||
                subset.allEqualValues
    }
    
    func calculateOutputProbabilities(subset: SubSet, tree: Tree) {
        let examples = subset.examples
        let examplesCount = Double(examples.count)
        let outputs = tree.trainingSet.outputs.count
        
        // count instances of each output class
        var probabilities = Array<Double>(count: outputs, repeatedValue: 0.0)
        for example in examples {
            probabilities[Int(example.output)] += 1.0
        }
        
        // divide through by the number of examples to produce probabilities
        var maxProb = 0.0
        var index = 0
        for i in 0..<outputs {
            let prob = probabilities[i] / examplesCount
            probabilities[i] = prob
            
            // keep track of highest probability output class
            if prob > maxProb {
                maxProb = prob
                index = i
            }
        }
        
        self.probabilities = probabilities
        self.outputIndex = index
    }
    
    func randomlySplit(subset: SubSet, tree: Tree) {
        var ranges = subset.featureRanges()
        var indexes = [Int](0..<ranges.count)
        
        // randomly select features to score
        for _ in 0..<(ranges.count - tree.numFeatures) {
            let index = lrand48() % indexes.count
            indexes.removeAtIndex(index)
            ranges.removeAtIndex(index)
        }
        
        // create random splits for each selected feature
        var splits = [Split]()
        for (index, range) in zip(indexes, ranges) {
            splits.append(Split(index: index, range: range, trainingSet: tree.trainingSet))
        }
        
        // collect subsets for each potential split
        for example in subset.examples {
            for split in splits {
                split.addExample(example)
            }
        }
        
        // score each split using weighted information gain as outlined
        // in `Extremely randomized trees'
        let Hc = subset.entropy()
        for split in splits {
            let Hs = split.entropy()
            split.score = (2 * (Hc - Hs)) / (Hc + Hs)
        }
        
        // pick the highest scoring split and construct children
        guard let maxSplit = splits.maxElement({ $0.score < $1.score }) else { fatalError() }
        splits.removeAll() // release memory before constructing more of the tree
        splitIndex = maxSplit.index
        splitValue = maxSplit.value
        left = Node()
        right = Node()
        
        // recursively continue constructing the tree
        left!.split(maxSplit.left, tree: tree)
        right!.split(maxSplit.right, tree: tree)
    }
}
