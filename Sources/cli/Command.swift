import Foundation
import SwiftForest
import SwiftCLI

class Command: OptionCommandType {
    var commandName: String
    var commandShortDescription: String
    var commandSignature: String

    init(commandName: String, commandShortDescription: String, commandSignature: String) {
        self.commandName = commandName
        self.commandShortDescription = commandShortDescription
        self.commandSignature = commandSignature
    }

    // training data - all commands can load training files
    var loadTrainingDataTask = LoadDataTask(outputsPresent: true, shuffle: true, delegate: LoadDataTaskDelegate(
        before: {
            print("Reading training files...")
        },

        after: {(rowSet, model) in
            print("Features: \(model.features)") // print so the result of -x can be seen
            print("Output classes: \(model.outputs)")
        }
    ))

    // training task - all commands can train classifiers
    var trainingTask = TrainingTask(delegate: TrainingTaskDelegate(
        before: {(task) in
            print("\nTraining forest of \(task.forestSize) trees")
        },

        after: {(classifier, duration) in
            let tree = (classifier as! Forest).trees[0] as! TrainableTree
            print("\nFinished, training took \(duration)")
            print("Tree depth: \(tree.maxDepth), leaves: \(tree.numLeaves)")
        }
    ))

    // classification data - classify and test commands use this task
    // the "--remove-features" flag applies to both load training data and load classifier data
    // tasks, so to avoid overriding the flag definition in subclasses the task is defined here.
    var loadClassifierDataTask = LoadDataTask(outputsPresent: false, shuffle: false, delegate: LoadDataTaskDelegate(
        before: {
            print("\nReading input files...")
        },

        after: {(_, _) in}
    ))

    // quiet mode - prevents delegates from printing
    var quietMode = false

    func setupOptions(options: Options) {
        options.onFlags(["-q", "--quiet"], usage: "Quiet mode - prevents status output from printing") {(flag) in
            self.quietMode = true
        }

        options.onKeys(["-r", "--forest"], usage: "Number trees to generate (default: 100)", valueSignature: "trees") {(key, value) in
            if let trees = Int(value) {
                self.trainingTask.forestSize = trees
            }
        }

        options.onKeys(["-e", "--features"], usage: "Number of features to compare at each tree node (default: sqrt(|features|))", valueSignature: "features") {(key, value) in
            if let features = Int(value) {
                self.trainingTask.pickFeatures = features
            }
        }

        options.onKeys(["-m", "--min-examples"], usage: "Minimum number of training examples a node must have to split (default: 2)", valueSignature: "examples") {(key, value) in
            if let examples = Int(value) {
                self.trainingTask.minExamples = examples
            }
        }

        options.onKeys(["-s", "--random-seed"], usage: "Seed for the random number generator (default: 1)", valueSignature: "seed") {(key, value) in
            if let seed = Int(value) {
                self.trainingTask.randomSeed = seed
            }
        }

        options.onKeys(["-x", "--remove-features"], usage: "Comma separated list of feature indexes to remove before training", valueSignature: "indexes") {(key, indexes) in
            let indexes = indexes.componentsSeparatedByString(",").map { Int($0)! }
            self.loadTrainingDataTask.removeFeatures = indexes
            self.loadClassifierDataTask.removeFeatures = indexes
        }
    }

    func setup(arguments: CommandArguments) {
        trainingTask.loadDataTask = loadTrainingDataTask
        if loadTrainingDataTask.paths == nil {
            loadTrainingDataTask.paths = arguments.requiredCollectedArgument("training_file")
        }

        if quietMode {
            loadTrainingDataTask.delegate = nil
            trainingTask.delegate = nil
        }
    }

    func execute(arguments: CommandArguments) throws  {
        fatalError("execute must be overriden")
    }
}
