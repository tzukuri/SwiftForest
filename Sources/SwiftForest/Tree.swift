import Foundation

// Implementation of:
// Extremely randomized trees
// http://www.montefiore.ulg.ac.be/%7Eernst/uploads/news/id63/extremely-randomized-trees.pdf


// ---------------------------------------
// protocols
// ---------------------------------------
public protocol Classifier {
    func distribution(row: Row) -> Distribution
    func classify(row: Row) -> Int
    var model: Model { get }
}

public protocol TrainableClassifier: Classifier {
    func train(trainingSet: TrainingSet)
    var delegate: ClassifierDelegate? { get set }
}

public protocol ClassifierDelegate {
    mutating func progressSteps(count: Int)
    mutating func step()
    mutating func finish()
}



// ---------------------------------------
// classification
// ---------------------------------------
public class Tree: Classifier, CustomStringConvertible {
    public var model: Model
    public var root: Node?

    public init(model: Model) {
        self.model = model
    }

    public var description: String {
        if let root = self.root {
            return root.description
        } else {
            fatalError("Cannot generate description for tree with no root node")
        }
    }

    internal func findLeafNode(row: Row) -> Node {
        guard let root = self.root else {
            fatalError("Cannot call classify on tree with no root node")
        }
        
        // walk the tree until we reach a leaf
        var node = root
        while !node.leaf {
            guard let index = node.splitIndex, value = node.splitValue else {
                fatalError("Cannot classify on untrained node")
            }

            if row.values[index] < value {
                node = node.left!
            } else {
                node = node.right!
            }
        }

        return node
    }

    public func distribution(row: Row) -> Distribution {
        let node = findLeafNode(row)
        if let outputDistribution = node.outputDistribution {
            return outputDistribution
        } else {
            fatalError("Cannot classify on untrained node")
        }
    }

    public func classify(row: Row) -> Int {
        return distribution(row).max()
    }
}



// ---------------------------------------
// training
// ---------------------------------------
final public class TrainableTree: Tree, TrainableClassifier {
    public var delegate: ClassifierDelegate?
    public var trainingSet: TrainingSet!
    public var pickFeatures: Int!
    public var minExamples: Int

    // trees store their own random generator state
    public var xsubi: [UInt16] = [1, 1, 1]

    // breadth/depth tracking
    public var numLeaves = 0
    public var maxDepth = 0
    
    public init(model: Model, pickFeatures: Int? = nil, minExamples: Int = 2) {
        self.pickFeatures = pickFeatures
        self.minExamples = minExamples
        super.init(model: model)
    }

    public func seedRandomGenerator(seed: Int32) {
        xsubi[0] = 0x330e                 // same fixed lower short srand48 uses
        xsubi[1] = UInt16(seed & 0xffff)  // lower short
        xsubi[2] = UInt16(seed >> 16)     // upper short
    }
    
    public func train(trainingSet: TrainingSet) {
        delegate?.progressSteps(1)

        // the root node splits a SubSet covering all examples
        let allExamples = SubSet(examples: trainingSet.examples, model: model)
        self.trainingSet = trainingSet

        // reset breadth/depth tracking
        numLeaves = 0
        maxDepth = 0

        // default pickFeatures to sqrt(|features|)
        if pickFeatures == nil {
            pickFeatures = Int(sqrt(Double(model.numFeatures)))
        }

        // reset the root node on train so the tree can be reused
        let rootNode = TrainableNode(tree: self, depth: 0, subset: allExamples)
        rootNode.train()
        root = rootNode

        // there's only a single tree to build, so progress is 0.0 -> 1.0
        delegate?.step()
        delegate?.finish()
    }
}
