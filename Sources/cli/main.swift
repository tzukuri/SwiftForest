import Foundation
import SwiftForest
import SwiftCLI

CLI.setup(
    name: "SwiftForest",
    version: "1.0",
    description: "Swift Forest - random forest decision tree"
)

CLI.registerCommands([
    CVCommand(),
    HoldoutCommand()
])

CLI.go()
