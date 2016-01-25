import PackageDescription

let package = Package(
    name: "SwiftForest",
    targets: [
        Target(
            name: "SwiftForest"
        ),
        Target(
            name: "cli",
            dependencies: [
                .Target(name: "SwiftForest")
            ]
        )
    ]
)

