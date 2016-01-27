import Foundation

final public class Example {
    public var values: [Double]
    public var output: Int
    
    public init(values: [Double], output: Int) {
        self.values = values
        self.output = output
    }
}

final public class TrainingSet {
    public var examples = [Example]()
    public var features: [String]
    public var outputs: [String]
    
    public init(features: [String], outputs: [String]) {
        self.features = features
        self.outputs = outputs
    }

    public var numOutputs: Int {
        return outputs.count
    }

    public var numFeatures: Int {
        return features.count
    }
    
    public func addExample(values: [Double], output: Int) {
        assert(values.count == features.count)
        assert(output < outputs.count)
        assert(output >= 0)
        examples.append(Example(values: values, output: output))
    }

    public func cloneWithExamples(examples: [Example]) -> TrainingSet {
        let trainingSet = TrainingSet(features: features, outputs: outputs)
        trainingSet.examples = examples
        return trainingSet
    }
}

final public class SubSet {
    public var examples = [Example]()
    public var outputCounts: [Int]
    
    public init(examples: [Example], trainingSet: TrainingSet) {
        self.outputCounts = Array<Int>(count: trainingSet.numOutputs, repeatedValue: 0)

        for example in examples {
            self.append(example)
        }
    }
    
    public convenience init(trainingSet: TrainingSet) {
        self.init(examples: [], trainingSet: trainingSet)
    }
    
    public var count: Int {
        return examples.count
    }
    
    /// True when all examples have the same output class
    public var allEqualOutputs: Bool {
        let zeroOutputs = outputCounts.filter({ $0 == 0 }).count
        return zeroOutputs == (outputCounts.count - 1)
    }
    
    /// True when all examples have the same values
    public var allEqualValues: Bool {
        guard let values = examples.first?.values else { return true }
        return !examples.dropFirst().contains { $0.values != values }
    }

    public func append(example: Example) {
        outputCounts[example.output] += 1
        examples.append(example)
    }
    
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

    public func entropy() -> Double {
        let totalCount = Double(examples.count)
        var totalEntropy = 0.0
        
        for count in outputCounts {
            let prob = Double(count) / totalCount
            let entropy = prob * log2(prob)
            totalEntropy -= entropy
        }
        
        return totalEntropy
    }
}

