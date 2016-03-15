import SwiftForest
import Foundation

internal class CSV {
    // processes a CSV file with this format:
    // line 0: #feature1,feature2,feature3,...,output_class
    // lin1 1: #output_class1,output_class2,...
    static func read(path: String, inout rowSet: RowSet!, inout model: Model!, outputsPresent: Bool) {
        // attempt to read the entire file into memory
        guard let data = try? NSString(contentsOfFile: path, encoding: NSUTF8StringEncoding) else {
            fatalError("Exception reading CSV file")
        }

        var trainingSet: TrainingSet!
        if let rowSet = rowSet {
            if outputsPresent {
                trainingSet = rowSet as! TrainingSet
            }
        } else {
            if outputsPresent {
                trainingSet = TrainingSet()
                rowSet = trainingSet
            } else {
                rowSet = RowSet()
            }
        }

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
        var features = featureLabelsLine.componentsSeparatedByString(",")
        if outputsPresent {
            features.removeLast()
        }

        // output labels are separated by commas
        let outputs = outputLabelsLine.componentsSeparatedByString(",")

        // create a new model, or ensure the file matches the existing model
        if let model = model {
            try! model.merge(features: features, outputs: outputs)
        } else {
            model = Model(features: features, outputs: outputs)
        }

        // the remaining lines are classification/training examples
        var outputClass: String!

        for line in lines {
            var parts = line.componentsSeparatedByString(",")
            if outputsPresent {
                outputClass = parts.removeLast()
            }

            let values = parts.map { Double($0)! }

            if outputsPresent {
                trainingSet.addExample(values, outputClass: outputClass)
            } else {
                rowSet.addRow(values)
            }
        }
    }
}
