import Foundation
import SwiftForest
import SwiftCLI

class HoldoutCommand: OptionCommandType {
    let commandName = "holdout"
    let commandShortDescription = "Trains a new forest on training input and measures accuracy on holdout input"
    let commandSignature = "<training_file> <holdout_file>"

    var removeFeatures: [Int]? = nil
    var pickFeatures: Int? = nil
    var forestSize = 100
    var minExamples = 2
    var randomSeed = 1

    func setupOptions(options: Options) {
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
        // training input
        print("Reading training file...")
        let (trainingModel, trainingSet) = CSV.read(
            arguments.requiredArgument("training_file")
        )

        if let indexes = removeFeatures {
            trainingSet.removeFeatures(indexes)
            trainingModel.removeFeatures(indexes)
        }

        // print labels so the result of --remove-features can be seen
        print("Features: \(trainingModel.features)")
        print("Output classes: \(trainingModel.outputs)")

        // prepare classifier
        print("\nTraining forest of \(forestSize) trees")
        trainingSet.shuffleExamples()
        let classifier = TrainableForest(
            model: trainingModel,
            size: forestSize,
            pickFeatures: pickFeatures,
            minExamples: minExamples,
            randomSeed: randomSeed,
            delegate: ClassifierProgress()
        )

        // benchmark forest creation
        var start = NSDate()
        classifier.train(trainingSet)
        var duration = start.timeIntervalSinceNow * -1

        let tree = classifier.trees[0] as! TrainableTree
        print("\nFinished, training took \(duration)")
        print("Tree depth: \(tree.maxDepth), leaves: \(tree.numLeaves)")


        // testing input
        print("\nReading holdout file...")
        let (holdoutModel, holdoutSet) = CSV.read(
            arguments.requiredArgument("holdout_file")
        )

        if let indexes = removeFeatures {
            holdoutSet.removeFeatures(indexes)
            holdoutModel.removeFeatures(indexes)
        }

        // benchmark testing
        start = NSDate()
        let testSet = TestSet(trainingSet: trainingSet, testExamples: holdoutSet.examples[0..<holdoutSet.examples.count])
        let accuracy = testSet.score(classifier)
        duration = start.timeIntervalSinceNow * -1

        print("Finished, testing took \(duration)")
        print("Model accuracy: \(accuracy)")
    }
}
