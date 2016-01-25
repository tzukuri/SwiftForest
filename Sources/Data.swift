import Foundation

class Example {
    var values: [Double]
    var output: UInt
    
    init(values: [Double], output: UInt) {
        self.values = values
        self.output = output
    }
}

public class TrainingSet {
    var examples = [Example]()
    var features: [String]
    var outputs: [String]
    
    public init(features: [String], outputs: [String]) {
        self.features = features
        self.outputs = outputs
    }
    
    public func addExample(values: [Double], output: UInt) {
        assert(values.count == features.count)
        assert(Int(output) < outputs.count)
        examples.append(Example(values: values, output: output))
    }
}

class SubSet {
    var examples: [Example]
    
    init(examples: [Example]) {
        self.examples = examples
    }
    
    init() {
        self.examples = []
    }
    
    var count: Int {
        return examples.count
    }
    
    var allEqualOutputs: Bool {
        guard let output = examples.first?.output else { return true }
        return !examples.dropFirst().contains { $0.output != output }
    }
    
    var allEqualValues: Bool {
        guard let values = examples.first?.values else { return true }
        return !examples.dropFirst().contains { $0.values != values }
    }
    
    func featureRanges() -> [(min: Double, max: Double)] {
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
    
    func outputCounts() -> [UInt: Int] {
        var counts = [UInt: Int]()
        for example in examples {
            counts[example.output] = (counts[example.output] ?? 0) + 1
        }
        
        return counts
    }
    
    func entropy() -> Double {
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

