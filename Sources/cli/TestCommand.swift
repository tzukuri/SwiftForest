import Foundation
import SwiftForest
import SwiftCLI

class TestCommand: ClassifyCommand {
    convenience init() {
        self.init(
            commandName: "test",
            commandShortDescription: "Classify input files on a new or existing classifier and calculate the classification accuracy",
            commandSignature: "<input_file> ..."
        )
    }

    // test scoring task
    var testTask = TestTask()

    override func setup(arguments: CommandArguments) {
        super.setup(arguments)
        loadClassifierDataTask.outputsPresent = true
        testTask.classificationTask = classificationTask
    }

    override func execute(arguments: CommandArguments) throws  {
        setup(arguments)
        let accuracy = testTask.run()
        print("Accuracy: \(accuracy)")
    }
}
