import Foundation
import SwiftForest
import SwiftCLI
import Progress

struct CrossFoldProgress: CrossFoldValidationDelegate {
    func validationProgress(fold: Int) {
        print("\n# fold \(fold)")
    }
}

struct ClassifierProgress: ClassifierDelegate {
    var bar: ProgressBar?

    mutating func trainingWillStartWithSteps(count: Int) {
        bar = ProgressBar(count: count)
    }

    mutating func trainingDidCompleteStep() {
        bar?.next()
    }

    mutating func trainingDidFinish() {
        bar?.next()
    }
}

class TestCommand: OptionCommandType {
    let commandName = "test"
    let commandShortDescription = "Trains a new forest on an input file and measures accuracy"
    let commandSignature = "<training_file>"

    var testingFolds = 10
    var trainingFolds = 10
    var forestSize = 100
    var numFeatures: Int? = nil
    var minExamples = 2
    var randomSeed = 1

    func setupOptions(options: Options) {
        options.onKeys(["-f", "--training-folds"], usage: "Number of training folds to generate (default: 10)", valueSignature: "folds") {(key, value) in
            if let folds = Int(value) {
                self.trainingFolds = folds
            }
        }

        options.onKeys(["-t", "--testing-folds"], usage: "Number of folds to test (default: 10)", valueSignature: "folds") {(key, value) in
            if let folds = Int(value) {
                self.testingFolds = folds
            }
        }

        options.onKeys(["-r", "--forest"], usage: "Number trees to generate (default: 100)", valueSignature: "trees") {(key, value) in
            if let trees = Int(value) {
                self.forestSize = trees
            }
        }

        options.onKeys(["-e", "--features"], usage: "Number of features to compare at each tree node (default: sqrt(|features|))", valueSignature: "features") {(key, value) in
            if let features = Int(value) {
                self.numFeatures = features
            }
        }

        options.onKeys(["-m", "--min-examples"], usage: "Minimum number of training examples a node must have to split (default: 2)", valueSignature: "examples") {(key, value) in
            if let examples = Int(value) {
                self.minExamples = examples
            }
        }

        options.onKeys(["-s", "--random-seed"], usage: "Seed for the random number generator (default: 1)", valueSignature: "seed") {(key, value) in
            if let seed = Int(value) {
                self.randomSeed = seed
            }
        }
    }

    func execute(arguments: CommandArguments) throws  {
        print("Reading training file...")
        let trainingSet = CSV.read(
            arguments.requiredArgument("training_file")
        )

        print("Creating \(trainingFolds) folds...")
        let folds = CrossFoldValidation(
            trainingSet: trainingSet,
            folds: trainingFolds,
            delegate: CrossFoldProgress()
        )

        print("Forest of \(forestSize) trees per fold")
        let classifier = Forest(
            size: forestSize,
            numFeatures: numFeatures,
            minExamples: minExamples,
            randomSeed: randomSeed,
            delegate: ClassifierProgress()
        )

        // train and score folds
        print("Training and testing \(testingFolds) folds")
        let start = clock()
        let accuracy = folds.score(classifier, maxFolds: testingFolds)

        // print total testing duration and average fold accuracy
        let duration = Double(clock() - start)
        print("\nFinished, training took \(duration / Double(CLOCKS_PER_SEC))")
        print("Model accuracy: \(accuracy)")
    }
}
