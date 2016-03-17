import Foundation
import SwiftForest
import SwiftCLI

class TrainCommand: Command {
    convenience init() {
        self.init(
            commandName: "train",
            commandShortDescription: "Trains a new classifier on input files",
            commandSignature: "<training_file> ..."
        )
    }

    var outputPath: String!

    override func setupOptions(options: Options) {
        super.setupOptions(options)
        
        options.onKeys(["-o", "--output"], usage: "Path where the trained classifier will be written", valueSignature: "path") {(key, path) in
            self.outputPath = path
        }
    }

    override func execute(arguments: CommandArguments) throws  {
        setup(arguments)

        guard let path = outputPath else {
            fatalError("Output path was not specified")
        }

        let classifier = trainingTask.run()
        let forest = classifier as! Forest
        ForestSerialiser(forest: forest).write(path)
    }
}
