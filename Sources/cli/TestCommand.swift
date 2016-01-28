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

    var removeFeatures: [Int]? = nil
    var pickFeatures: Int? = nil
    var testingFolds = 10
    var trainingFolds = 10
    var forestSize = 100
    var minExamples = 2
    var randomSeed = 1

    func setupOptions(options: Options) {
        options.onKeys(["-f", "--training-folds"], usage: "Number of training folds to generate (default: 10)", valueSignature: "training_folds") {(key, value) in
            if let folds = Int(value) {
                self.trainingFolds = folds
            }
        }

        options.onKeys(["-t", "--testing-folds"], usage: "Number of folds to test (default: 10)", valueSignature: "testing_folds") {(key, value) in
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
                self.pickFeatures = features
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

        options.onKeys(["-x", "--remove-features"], usage: "Comma separated list of feature indexes to remove before training", valueSignature: "indexes") {(key, indexes) in
            self.removeFeatures = indexes.componentsSeparatedByString(",").map { Int($0)! }
        }
    }

    func execute(arguments: CommandArguments) throws  {
        print("Reading training file...")
        let (model, trainingSet) = CSV.read(
            arguments.requiredArgument("training_file")
        )

        if let indexes = removeFeatures {
            trainingSet.removeFeatures(indexes)
            model.removeFeatures(indexes)
        }

        print("Features: \(model.features)")
        print("Output classes: \(model.outputs)")

        print("Creating \(trainingFolds) training folds...")
        trainingSet.shuffleExamples()
        let folds = CrossFoldValidation(
            trainingSet: trainingSet,
            folds: trainingFolds,
            delegate: CrossFoldProgress()
        )

        print("\nForest of \(forestSize) trees per fold")
        let classifier = TrainableForest(
            model: model,
            size: forestSize,
            pickFeatures: pickFeatures,
            minExamples: minExamples,
            randomSeed: randomSeed,
            delegate: ClassifierProgress()
        )

        // train and score folds
        print("Testing \(testingFolds) folds on \(trainingSet.examples.count) examples")
        let start = NSDate()
        let accuracy = folds.score(classifier, maxFolds: testingFolds)

        // print total testing duration and average fold accuracy
        let duration = start.timeIntervalSinceNow * -1
        let tree = classifier.trees[0] as! TrainableTree
        print("\nFinished, training took \(duration)")
        print("Tree depth: \(tree.maxDepth), leaves: \(tree.numLeaves)")
        print("Model accuracy: \(accuracy)")
    }
}
