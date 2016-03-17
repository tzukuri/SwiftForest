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
    ],

    dependencies: [
        .Package(
            url: "https://github.com/willcannings/SwiftCLI.git",
            majorVersion: 1
        ),
        .Package(
            url: "https://github.com/tzukuri/FetchPack.git",
            majorVersion: 0
        ),
        .Package(
            url: "https://github.com/jkandzi/Progress.swift",
            majorVersion: 0
        )
    ]
)

