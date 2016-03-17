import Foundation
import FetchPack

// ---------------------------------------
// model
// ---------------------------------------
final public class Model: Serialisable {
    public var serialiserType: String { return "Model" }

    enum ModelError: ErrorType {
        case UnmatchedFeatures
    }

    public var features: [String]
    public var outputs: [String]

    public init(features: [String], outputs: [String]) {
        self.features = features
        self.outputs = outputs
    }

    public convenience init() {
        self.init(features: [], outputs: [])
    }

    public init(deserialiser: FetchPackDeserialiser) {
        self.features = deserialiser.read()
        self.outputs = deserialiser.read()
    }

    public func serialise(serialiser: FetchPackSerialiser) {
        serialiser.append(features)
        serialiser.append(outputs)
    }

    public var numOutputs: Int {
        return outputs.count
    }

    public var numFeatures: Int {
        return features.count
    }

    public func removeFeatures(indexes: [Int]) {
        for (i, index) in indexes.enumerate() {
            features.removeAtIndex(index - i)
        }
    }

    public func merge(features otherFeatures: [String], outputs otherOutputs: [String]) throws {
        if otherFeatures != features {
            throw ModelError.UnmatchedFeatures
        }

        for outputClass in otherOutputs {
            if outputs.indexOf(outputClass) == nil {
                outputs.append(outputClass)
            }
        }
    }
}



// ---------------------------------------
// probability distribution
// ---------------------------------------
final public class Distribution {
    public var probabilities: [Double]
    public var instances = 0.0
    public var finalised = false
    public let count: Int

    // memoised functions
    internal var _max: Int? = nil

    public init(count: Int) {
        self.probabilities = Array<Double>(count: count, repeatedValue: 0.0)
        self.count = count
    }

    /// Increment the count of one of the classes by 1
    public func increment(index: Int) {
        probabilities[index] += 1.0
        instances += 1.0
    }

    /// Indicate counting has finished and prepare the probabilities array
    public func finalise() {
        if finalised { return }
        finalised = true

        for i in 0..<probabilities.count {
            probabilities[i] /= instances
        }
    }

    /// Returns the index of the class with the highest probability
    public func max() -> Int {
        if let max = _max { return max }
        finalise()

        var maxProb  = probabilities[0]
        var maxIndex = 0
        
        for (i, probability) in probabilities.enumerate() {
            if probability > maxProb {
                maxProb = probability
                maxIndex = i
            }
        }

        _max = maxIndex
        return maxIndex
    }

    /// Calculates the information entropy over the probabilities
    public func entropy() -> Double {
        var totalEntropy = 0.0
        finalise()
        
        for probability in probabilities {
            if probability == 0.0 { continue }
            totalEntropy -= (probability * log2(probability))
        }
        
        return totalEntropy
    }
}



// ---------------------------------------
// classification
// ---------------------------------------
/// Single row of data. Rows are used during training, testing,
/// and classification.
public class Row {
    public var values: [Double]
    public init(values: [Double]) {
        self.values = values
    }
}


/// Collection of rows to use during classification.
public class RowSet: SequenceType {
    public typealias Generator = IndexingGenerator<[Row]>
    public var rows: [Row]

    public var count: Int {
        return rows.count
    }

    public init(rows: [Row]) {
        self.rows = rows
    }

    public convenience init() {
        self.init(rows: [])
    }

    public func generate() -> RowSet.Generator {
        return rows.generate()
    }

    public func addRow(values: [Double]) {
        rows.append(
            Row(values: values)
        )
    }

    public func shuffle() {
        // ensure shuffle is consistent
        srand48(1)

        // can't shuffle 0-1 item sets
        let count = rows.count
        if count < 2 { return }

        for i in 0..<(count - 1) {
            let j = (lrand48() % (count - i)) + i
            guard i != j else { continue }
            swap(&rows[i], &rows[j])
        }
    }

    public func removeFeatures(indexes: [Int]) {
        for row in rows {
            for (i, index) in indexes.enumerate() {
                row.values.removeAtIndex(index - i)
            }
        }
    }
}



// ---------------------------------------
// training
// ---------------------------------------
/// Single training example. `output' is the index of the output
/// class and must be within 0..<outputs.count
final public class Example: Row {
    public var outputClass: String
    public var output: Int!
    
    public init(values: [Double], outputClass: String) {
        self.outputClass = outputClass
        super.init(values: values)
    }
}


/// Collection of examples to use during training/testing
final public class TrainingSet: RowSet {
    public var examples: [Example] {
        return rows as! [Example]
    }

    public func addExample(values: [Double], outputClass: String) {
        rows.append(
            Example(values: values, outputClass: outputClass)
        )
    }

    public func setOutputIndexes(model: Model) {
        for example in examples {
            if let index = model.outputs.indexOf(example.outputClass) {
                example.output = index
            } else {
                fatalError("Unknown example output class: \(example.outputClass)")
            }
        }
    }
}


/// Reduced set of examples from an initial training set. Nodes in a tree
/// train on subsets which are created by splitting higher level subsets
/// on randomly selected split points.
final public class SubSet {
    public var examples = [Example]()
    public var outputDistribution: Distribution
    
    public init(examples: [Example], model: Model) {
        self.outputDistribution = Distribution(count: model.numOutputs)
        for example in examples {
            self.append(example)
        }
    }
    
    public convenience init(model: Model) {
        self.init(examples: [], model: model)
    }
    
    public var count: Int {
        return examples.count
    }
    
    /// True when all examples have the same output class
    public var allEqualOutputs: Bool {
        let zeroOutputs = outputDistribution.probabilities.filter({ $0 == 0.0 }).count
        return zeroOutputs == (outputDistribution.count - 1)
    }
    
    /// True when all examples have the same values
    public var allEqualValues: Bool {
        guard let values = examples.first?.values else { return true }
        return !examples.dropFirst().contains { $0.values != values }
    }

    /// Append a new example to the subset and increment `outputCounts'
    public func append(example: Example) {
        outputDistribution.increment(example.output)
        examples.append(example)
    }

    public func finalise() {
        outputDistribution.finalise()
    }
    
    /// Returns an array of (min, max) tuples for each feature in the model, i.e
    /// where min in the smallest observed value of a feature in this subset
    public func featureRanges() -> [(min: Double, max: Double)] {
        var ranges = [(min: Double, max: Double)]()
        for value in examples.first!.values {
            ranges.append((min: value, max: value))
        }
        
        for example in examples {
            for (i, value) in example.values.enumerate() {
                if value < ranges[i].min { ranges[i].min = value }
                if value > ranges[i].max { ranges[i].max = value }
            }
        }
        
        return ranges
    }

    /// Calculates the information entropy over the subset's output probabilities
    public func entropy() -> Double {
        return outputDistribution.entropy()
    }
}

