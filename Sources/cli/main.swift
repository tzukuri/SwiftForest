import Foundation
import SwiftForest

let ts = TrainingSet(features: ["a", "b"], outputs: ["f", "t"])
ts.addExample([0, 0], output: 0)
ts.addExample([0, 1], output: 1)
ts.addExample([1, 0], output: 1)
ts.addExample([1, 1], output: 0)

let f = Forest(size: 100)
f.train(ts)

print(f.classify([0, 0]))
print(f.classify([0, 1]))
print(f.classify([1, 0]))
print(f.classify([1, 1]))
