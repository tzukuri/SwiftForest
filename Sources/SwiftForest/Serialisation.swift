import Foundation
import FetchPack

// ---------------------------------------
// data types
// ---------------------------------------
public struct SerialisedNode {
    var splitIndex: UInt8
    var splitValue: Double
    var left: UInt32
    var right: UInt32

    init(splitIndex: UInt8, splitValue: Double) {
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
    public var leafNodes = [Double]()

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
        serialiser.appendRawArray(&leafNodes)
    }

    private func serialiseInternal(node: Node) -> NodePointer {
        let index = internalNodes.count
        internalNodes.append(
            SerialisedNode(
                splitIndex: UInt8(node.splitIndex!),
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
            leafNodes.append(probability)
        }
        
        return NodePointer.LeafNode(index: leafNodes.count / probabilities.count)
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
        serialiser.append(serialisedTrees)
    }

    required public init(deserialiser: FetchPackDeserialiser) {
        fatalError("ForestSerialiser cannot be deserialised")
    }

    public func write(path: String) {
        let writer = PackFileWriter(path: path, type: "swFo", format: 1)
        writer.serialise(self)
    }
}
