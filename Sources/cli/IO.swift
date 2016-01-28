import SwiftForest
import Foundation

internal class CSV {
    // processes a CSV file with this format:
    // line 0: #feature1,feature2,feature3,...,output_class
    // lin1 1: #output_class1,output_class2,...
    static func read(path: String) -> (Model, TrainingSet) {
        // try and read the entire file into memory
        guard let data = try? NSString(contentsOfFile: path, encoding: NSUTF8StringEncoding) else {
            fatalError("Exception reading CSV file")
        }

        let trainingSet = TrainingSet()
        let model = Model()

        // split into lines and ignore empty lines
        var lines = data.componentsSeparatedByString("\n").filter { $0 != "" }
        
        // the first 2 lines define the feature (column) labels and output labels
        var featureLabelsLine = lines.removeFirst()
        if featureLabelsLine.hasPrefix("#") {
            featureLabelsLine.removeAtIndex(featureLabelsLine.startIndex)
        }

        var outputLabelsLine = lines.removeFirst()
        if outputLabelsLine.hasPrefix("#") {
            outputLabelsLine.removeAtIndex(outputLabelsLine.startIndex)
        }

        // feature labels are separated by commas. the final feature label is the
        // output class and can be ignored
        model.features = featureLabelsLine.componentsSeparatedByString(",")
        model.features.removeLast()

        // output labels are separated by commas
        model.outputs = outputLabelsLine.componentsSeparatedByString(",")

        // the remaining lines are training examples
        for line in lines {
            var parts = line.componentsSeparatedByString(",")
            let output = Int(parts.removeLast())!
            let values = parts.map { Double($0)! }
            trainingSet.addExample(values, output: output)
        }

        return (model, trainingSet)
    }
}
