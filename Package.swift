// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OnDeviceRAGKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        // Library only — the runnable demo lives in a separate repo
        // (on-device-rag-kit-demo-app) that consumes this package by its
        // published git URL, the way any real external consumer would.
        .library(name: "OnDeviceRAGKit", targets: ["OnDeviceRAGKit"])
    ],
    targets: [
        .target(name: "OnDeviceRAGKit"),
        .testTarget(
            name: "OnDeviceRAGKitTests",
            dependencies: ["OnDeviceRAGKit"]
        ),
    ]
)
