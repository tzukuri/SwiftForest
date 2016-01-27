import SwiftForest
import Foundation

internal class CSV {
    // processes a CSV file with this format:
    // line 0: #feature1,feature2,feature3,...,output_class
    // lin1 1: #output_class1,output_class2,...
    static func read(path: String) -> TrainingSet {
        // try and read the entire file into memory
        guard let data = try? NSString(contentsOfFile: path, encoding: NSUTF8StringEncoding) else {
            fatalError("Exception reading CSV file")
        }

        // split into lines and ignore empty lines
        var lines = data.componentsSeparatedByString("\n").filter { $0 != "" }
        
        // the first 2 lines define the feature (column) labels and output labels
        var featureLabelsLine = lines.removeFirst()
        var outputLabelsLine = lines.removeFirst()

        // feature labels are separated by commas
        if featureLabelsLine.hasPrefix("#") {
            featureLabelsLine.removeAtIndex(featureLabelsLine.startIndex)
        }

        var featureLabels = featureLabelsLine.componentsSeparatedByString(",")

        // the final feature label is the output class and can be ignored
        featureLabels.removeLast()

        // output labels are separated by commas
        if outputLabelsLine.hasPrefix("#") {
            outputLabelsLine.removeAtIndex(outputLabelsLine.startIndex)
        }

        let outputLabels = outputLabelsLine.componentsSeparatedByString(",")

        // construct a new training set to start loading examples
        let ts = TrainingSet(features: featureLabels, outputs: outputLabels)

        for line in lines {
            var parts = line.componentsSeparatedByString(",")
            let output = Int(parts.removeLast())!
            let values = parts.map { Double($0)! }
            ts.addExample(values, output: output)
        }

        return ts
    }
}
