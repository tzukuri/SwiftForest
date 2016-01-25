import Foundation

// Implementation of:
// Extremely randomized trees
// http://www.montefiore.ulg.ac.be/%7Eernst/uploads/news/id63/extremely-randomized-trees.pdf

class Forest {
    var trees = [Tree]()
    
    init(size: Int, numFeatures: Int? = nil, minExamples: Int = 1, randomSeed: Int = 1) {
        for _ in 0..<size {
            self.trees.append(
                Tree(numFeatures: numFeatures, minExamples: minExamples, randomSeed: randomSeed)
            )
        }
    }
    
    func train(trainingSet: TrainingSet) {
        for tree in trees {
            tree.train(trainingSet)
        }
    }
    
    func classify(values: [Double]) -> Int {
        var counts = [Int: Int]()
        
        // classify example in each tree and count number of times
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

class Tree {
    var trainingSet: TrainingSet!
    var numFeatures: Int!
    let minExamples: Int
    let randomSeed: Int
    var root: Node
    
    init(numFeatures: Int? = nil, minExamples: Int = 1, randomSeed: Int = 1) {
        self.numFeatures = numFeatures
        self.minExamples = minExamples
        self.randomSeed = randomSeed
        self.root = Node()
    }
    
    func train(trainingSet: TrainingSet) {
        self.trainingSet = trainingSet
        if self.numFeatures == nil {
            self.numFeatures = Int(sqrt(Double(trainingSet.features.count)))
        }
        
        let allExamples = SubSet(examples: trainingSet.examples)
        root.split(allExamples, tree: self)
    }
    
    func classify(values: [Double]) -> Int {
        var node: Node = root
        
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

class Split {
    let index: Int
    let value: Double
    var score = 0.0
    let left = SubSet()
    let right = SubSet()
    
    init(index: Int, range: (min: Double, max: Double)) {
        let interval = range.max - range.min
        self.value = (drand48() * interval) + range.min
        self.index = index
    }
    
    func addExample(example: Example) {
        if example.values[index] < value {
            left.examples.append(example)
        } else {
            right.examples.append(example)
        }
    }
    
    func entropy() -> Double {
        let totalCount = Double(left.examples.count + right.examples.count)
        let Hl = left.entropy() * (Double(left.examples.count) / totalCount)
        let Hr = right.entropy() * (Double(right.examples.count) / totalCount)
        return Hl + Hr
    }
}

class Node {
    var splitIndex: Int? = nil          // feature index
    var splitValue: Double? = nil       // split point
    var probabilities: [Double]? = nil  // probability of each output class
    var outputIndex: Int?               // index of the highest probability output
    var left: Node?  = nil              // < split
    var right: Node? = nil              // >= split
    
    var leaf: Bool {
        return left == nil && right == nil
    }
    
    func split(subset: SubSet, tree: Tree) {
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
            let index = Int(arc4random_uniform(UInt32(indexes.count)))
            indexes.removeAtIndex(index)
            ranges.removeAtIndex(index)
        }
        
        // create random splits for each selected feature
        var splits = [Split]()
        for (index, range) in zip(indexes, ranges) {
            splits.append(Split(index: index, range: range))
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
