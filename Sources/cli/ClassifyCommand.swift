import Foundation
import SwiftForest
import SwiftCLI

class ClassifyCommand: Command {
    convenience init() {
        self.init(
            commandName: "classify",
            commandShortDescription: "Classify input files on a new or existing classifier",
            commandSignature: "<input_file> ..."
        )
    }

    // classifier
    var loadClassifierTask = LoadClassifierTask()
    var performTraining = false
    var loadClassifier = false

    // classification
    var classificationTask = ClassificationTask(delegate: ClassificationTaskDelegate(
        before: {
            print("\nClassifying...")
        },

        after: {(duration) in
            print("Classification took \(duration)")
        }
    ))
    
    override func setupOptions(options: Options) {
        super.setupOptions(options)

        // classifier
        options.onKeys(["-t", "--train"], usage: "Create a new classifier with training data from the specified file", valueSignature: "training_file") {(key, path) in
            self.loadTrainingDataTask.paths = [path]
            self.performTraining = true
        }

        options.onKeys(["-l", "--load"], usage: "Load an existing classifier stored in the specified file", valueSignature: "classifier_file") {(key, path) in
            self.loadClassifierTask.path = path
            self.loadClassifier = true
        }
    }

    override func setup(arguments: CommandArguments) {
        if performTraining && loadClassifier {
            fatalError("Cannot combine --train and --load options - you can only create or load a classifier")
        }

        // select classifier
        if performTraining {
            super.setup(arguments) // TrainCommand
            classificationTask.classifierProvider = trainingTask
        } else {
            classificationTask.classifierProvider = loadClassifierTask
        }

        // setup classification data
        loadClassifierDataTask.paths = arguments.requiredCollectedArgument("input_file")
        classificationTask.loadDataTask = loadClassifierDataTask

        if quietMode {
            loadClassifierDataTask.delegate = nil
            classificationTask.delegate = nil
        }
    }

    override func execute(arguments: CommandArguments) throws  {
        setup(arguments)
        let classifications = classificationTask.run()
        
        // print classification outputs to stdout
        for outputClass in classifications {
            print(outputClass)
        }
    }
}
