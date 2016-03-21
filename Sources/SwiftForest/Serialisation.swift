import Foundation
import FetchPack

// ---------------------------------------
// data types
// ---------------------------------------
public struct SerialisedNode {
    var splitIndex: UInt64
    var splitValue: Double
    var left: UInt32
    var right: UInt32

    init(splitIndex: UInt64, splitValue: Double) {
        self.splitIndex = splitIndex
        self.splitValue = splitValue
        self.left = 0
        self.right = 0
    }
}

// highest bit acts as a flag indicating the index is to
// an internal (not leaf) node
let internalFlag = UInt32(0x80000000)

// pointers can be to internal or leaf nodes. when serialised
// the most significant bit determines the enum type.
// TODO: determine whether NodePointer should accept/return 0
// based indexes so later code doesn't need to +/- 1
public enum NodePointer {
    case Nil
    case InternalNode(index: Int)
    case LeafNode(index: Int)

    public init(serialisedValue: UInt32) {
        if serialisedValue == 0 {
            self = .Nil
        } else if (serialisedValue & internalFlag) != 0 {
            let indexWithoutFlag = serialisedValue & ~internalFlag
            self = .InternalNode(index: Int(indexWithoutFlag))
        } else {
            self = .LeafNode(index: Int(serialisedValue))
        }
    }

    public var serialisedValue: UInt32 {
        switch self {
        case .Nil:
            return 0
        case .InternalNode(let index):
            return UInt32(index) | internalFlag
        case .LeafNode(let index):
            let index = UInt32(index)
            if (index & internalFlag) != 0 {
                fatalError("leaf node indexes cannot have the msb set")
            } else {
                return index
            }
        }
    }
}


// ---------------------------------------
// serialisers
// ---------------------------------------
public class TreeSerialiser: Serialisable {
    public var serialiserType: String { return "Tree" }
    public var internalNodes = [SerialisedNode]()
    public var leafNodeData = [Double]()
    public var leafNodeCount = 0

    public init(tree: Tree) {
        guard let root = tree.root else {
            fatalError("Cannot serialise untrained tree")
        }

        if root.leaf {
            serialiseLeaf(root)
        } else {
            serialiseInternal(root)
        }
    }

    required public init(deserialiser: FetchPackDeserialiser) {
        fatalError("TreeSerialiser cannot be deserialised")
    }

    public func serialise(serialiser: FetchPackSerialiser) {
        serialiser.appendRawArray(&internalNodes)
        serialiser.appendUInt64(leafNodeCount)
        serialiser.appendRawArray(&leafNodeData)
    }

    private func serialiseInternal(node: Node) -> NodePointer {
        let index = internalNodes.count
        internalNodes.append(
            SerialisedNode(
                splitIndex: UInt64(node.splitIndex!),
                splitValue: node.splitValue!
            )
        )

        if let left = node.left {
            let pointer = left.leaf ? serialiseLeaf(left) : serialiseInternal(left)
            internalNodes[index].left = pointer.serialisedValue
        }

        if let right = node.right {
            let pointer = right.leaf ? serialiseLeaf(right) : serialiseInternal(right)
            internalNodes[index].right = pointer.serialisedValue
        }

        // node indexes are 1 based (0 == nil or no link)
        return NodePointer.InternalNode(index: index + 1)
    }

    private func serialiseLeaf(node: Node) -> NodePointer {
        // store probabilities one after another in the leaf nodes array
        // e.g if there are 4 output classes, leaf node 2 (1 based index)
        // covers leafNodes[4..<8]
        let probabilities = node.outputDistribution!.probabilities
        for probability in probabilities {
            leafNodeData.append(probability)
        }

        // the last element in the flattened array is the index of the
        // max output class
        leafNodeData.append(Double(node.outputDistribution!.max()))
        
        leafNodeCount += 1
        return NodePointer.LeafNode(index: leafNodeCount)
    }
}

public class ForestSerialiser: Serialisable {
    public var serialiserType: String { return "Forest" }
    public var serialisedTrees = [TreeSerialiser]()
    public var forest: Forest

    public init(forest: Forest) {
        self.forest = forest

        for tree in forest.trees {
            self.serialisedTrees.append(
                TreeSerialiser(tree: tree)
            )
        }
    }

    public func serialise(serialiser: FetchPackSerialiser) {
        serialiser.append(forest.model)
        let objects = serialisedTrees.map { return $0 as Serialisable }
        serialiser.append(objects)
    }

    required public init(deserialiser: FetchPackDeserialiser) {
        fatalError("ForestSerialiser cannot be deserialised")
    }

    public func write(path: String) {
        let writer = PackFileWriter(path: path, type: "swFo", format: 1)
        writer.serialise(self)
    }
}



// ---------------------------------------
// deserialisers
// ---------------------------------------
public struct SerialisedDistribution: DistributionType {
    private let data: UnsafeMutableBufferPointer<Double>
    public  let count: Int
    private let start: Int

    public init(data: UnsafeMutableBufferPointer<Double>, start: Int, count: Int) {
        self.data = data
        self.start = start
        self.count = count
    }

    public subscript(index: Int) -> Double {
        return data[start + index]
    }

    public func max() -> Int {
        return Int(data[start + count])
    }
}

public class SerialisedTree: Serialisable, Classifier {
    public  var serialiserType: String { return "Tree" }
    public  let internalNodes: UnsafeMutableBufferPointer<SerialisedNode>
    private let leafNodeData: UnsafeMutableBufferPointer<Double>
    public  var leafNodes = [SerialisedDistribution]()
    private let leafNodeCount: Int
    public  var model: Model

    required public init(deserialiser: FetchPackDeserialiser) {
        self.internalNodes = deserialiser.readRawBufferPointer()
        self.leafNodeCount = deserialiser.readUInt64()
        self.leafNodeData = deserialiser.readRawBufferPointer()
        self.model = Model()
    }

    public func serialise(serialiser: FetchPackSerialiser) {
        fatalError("SerialisedTree cannot be serialised")
    }

    private func deserialiseLeafNodes() {
        // each serialised distribution stores a score for each
        // output, plus the index of the max output
        let count = model.outputs.count + 1

        for i in 0..<leafNodeCount {
            leafNodes.append(
                SerialisedDistribution(
                    data: leafNodeData,
                    start: i * count,
                    count: count - 1
                )
            )
        }
    }

    public func distribution(row: Row) -> DistributionType {
        guard internalNodes.count > 0 || leafNodes.count > 0 else {
            fatalError("Cannot call classify on tree with no root node")
        }

        // tree consists only of a leaf node
        if internalNodes.count == 0 {
            return leafNodes[0]
        }

        // walk the tree until we reach a leaf
        var node = internalNodes[0]
        while true {
            var index: UInt32

            if row.values[Int(node.splitIndex)] < node.splitValue {
                index = node.left
            } else {
                index = node.right
            }

            // pointers are 1 based
            let ptr = NodePointer(serialisedValue: index)
            switch ptr {
            case .Nil:
                fatalError("Serialised node points to null")
            case .InternalNode(let index):
                node = internalNodes[index - 1]
            case .LeafNode(let index):
                return leafNodes[index - 1]
            }
        }
    }
}

public class SerialisedForest: Serialisable, Classifier {
    public var serialiserType: String { return "Forest" }
    public var trees: [SerialisedTree]
    public var model: Model

    required public init(deserialiser: FetchPackDeserialiser) {
        self.model = deserialiser.readObject() as! Model
        self.trees = deserialiser.readObjectArray().map { $0 as! SerialisedTree }

        // model is a required property of a classifier, but is
        // only stored by the forest, not each tree
        for tree in self.trees {
            tree.model = self.model
            tree.deserialiseLeafNodes()
        }
    }

    public func serialise(serialiser: FetchPackSerialiser) {
        fatalError("ForestSerialiser cannot be serialised")
    }

    public static func read(path: String) -> SerialisedForest {
        registerSerialisationTypes()
        let reader = PackFileReader(path: path, type: "swFo", format: 1)
        return reader.deserialise() as! SerialisedForest
    }

    public func distribution(row: Row) -> DistributionType {
        let distribution = Distribution(count: model.numOutputs)
        
        // classify values in each tree and count number of times
        // each output class is selected
        for tree in trees {
            let output = tree.classify(row)
            distribution.increment(output)
        }

        // weight by total count to form probabilities
        distribution.finalise()
        return distribution
    }
}

private var registeredSerialisationTypes = false
private func registerSerialisationTypes() {
    if registeredSerialisationTypes { return }
    register("Forest", type: SerialisedForest.self)
    register("Tree", type: SerialisedTree.self)
    register("Model", type: Model.self)
    registeredSerialisationTypes = true
}
