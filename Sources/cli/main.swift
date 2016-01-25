import Foundation
import SwiftForest
import SwiftCLI

func trainModel(path: String) {
    print("Reading training file...")
    let ts = TrainingSet(features: ["a", "b"], outputs: ["f", "t"])
    // ts.addExample([0, 0], output: 0)
    // ...

    print("Training forest...")
    let start = clock()
    let f = Forest()
    f.train(ts)

    let duration = Double(clock() - start)
    print("Finished, training took \(duration / Double(CLOCKS_PER_SEC))")
}

CLI.setup(
    name: "SwiftForest",
    version: "1.0",
    description: "Swift Forest - random forest decision tree"
)

CLI.registerChainableCommand(commandName: "train")
    .withShortDescription("Trains a new forest on an input training file")
    .withSignature("<training_file>")
    .withExecutionBlock { (arguments) in
        trainModel(arguments.requiredArgument("training_file"))
    }

CLI.go()

