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
    public var examples: [Example]
    
    public init(examples: [Example]) {
        self.examples = examples
    }
    
    public init() {
        self.examples = []
    }
    
    public var count: Int {
        return examples.count
    }
    
    public var allEqualOutputs: Bool {
        guard let output = examples.first?.output else { return true }
        return !examples.dropFirst().contains { $0.output != output }
    }
    
    public var allEqualValues: Bool {
        guard let values = examples.first?.values else { return true }
        return !examples.dropFirst().contains { $0.values != values }
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
    
    public func outputCounts() -> [Int: Int] {
        var counts = [Int: Int]()
        for example in examples {
            counts[example.output] = (counts[example.output] ?? 0) + 1
        }
        
        return counts
    }
    
    public func entropy() -> Double {
        let totalCount = Double(examples.count)
        var totalEntropy = 0.0
        
        for (_, count) in outputCounts() {
            let prob = Double(count) / totalCount
            let entropy = prob * log2(prob)
            totalEntropy -= entropy
        }
        
        return totalEntropy
    }
}

