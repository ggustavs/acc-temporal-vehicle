-- ACC properties under closed-loop rollout from a single initial state.
import STL

-- QLL (Quantile Loss Logic): logsumexp-smoothed STL.
-- Reference: property-driven-ml/src/property_driven_ml/logics/qll.py.
qllP : Real
qllP = 5.0

eulerE : Real
eulerE = 2.718281828459045

-- Stable max-subtraction log-sum-exp: ln sum(exp a_i) = m + ln sum(exp(a_i - m)),
-- m = max a_i. Required for a finite gradient over the temporal fold; do not
-- revert to the plain exp/log form.
QLLLoss : DifferentiableTensorLogic
QLLLoss =
  { trueElement                = -1000000
  , falseElement               = 1000000
  , pointwiseNegation          = \x -> -x
  , pointwiseConjunction       = \{dims} x y ->
      let a = const qllP dims * x in
      let b = const qllP dims * y in
      let m = max a b in
      (m + log (const eulerE dims) (exp (a - m) + exp (b - m)))
        / const qllP dims
  , pointwiseDisjunction       = \{dims} x y ->
      let a = -(const qllP dims) * x in
      let b = -(const qllP dims) * y in
      let m = max a b in
      -(m + log (const eulerE dims) (exp (a - m) + exp (b - m)))
        / const qllP dims
  , pointwiseLessThan          = \{dims} x y ->
      let a = const qllP dims * (x - y) in
      let b = const qllP dims * (-(max (x - y) (y - x))) in
      let m = max a b in
      (m + log (const eulerE dims) (exp (a - m) + exp (b - m)))
        / const qllP dims
  , pointwiseLessEqualThan     = \x y -> x - y
  , pointwiseGreaterThan       = \{dims} x y ->
      let a = const qllP dims * (y - x) in
      let b = const qllP dims * (-(max (x - y) (y - x))) in
      let m = max a b in
      (m + log (const eulerE dims) (exp (a - m) + exp (b - m)))
        / const qllP dims
  , pointwiseGreaterEqualThan  = \x y -> y - x
  , pointwiseEqual             = \x y -> max (x - y) (y - x)
  , pointwiseNotEqual          = \x y -> -(max (x - y) (y - x))
  , reduceConjunction          = \{dims} e xs ->
      let m = reduceMax (qllP * e) (const qllP dims * xs) in
      (m + log eulerE
             (exp (qllP * e - m)
              + reduceAdd 0 (exp (const qllP dims * xs - const m dims))))
        / qllP
  , reduceDisjunction          = \{dims} e xs ->
      let m = reduceMax (-qllP * e) (const (-qllP) dims * xs) in
      -(m + log eulerE
              (exp (-qllP * e - m)
               + reduceAdd 0 (exp (const (-qllP) dims * xs - const m dims))))
        / qllP
  }

stateDim : Nat
stateDim = 6

obsDim : Nat
obsDim = 5

actDim : Nat
actDim = 1

T = 50

vSet : Real
vSet = 30.0

tGap : Real
tGap = 1.4

dDefault : Real
dDefault = 10.0

comfortMax : Real
comfortMax = 2.0

epsV : Real
epsV = 0.5

epsVGlobal : Real
epsVGlobal = 2.0

@network
controller : Tensor Real [obsDim] -> Tensor Real [actDim]

@dynamics
dynamics : Tensor Real [stateDim] -> Tensor Real [actDim]
        -> Tensor Real [stateDim]

@dataset
initState : Tensor Real [stateDim]

stateToObs : Tensor Real [stateDim] -> Tensor Real [obsDim]
stateToObs s = [vSet, tGap, s ! 4, s ! 0 - s ! 3, s ! 1 - s ! 4]

controllerOnState : Tensor Real [stateDim] -> Tensor Real [actDim]
controllerOnState s = controller (stateToObs s)

trajectory : Tensor Real [T, stateDim]
trajectory = rollout T controllerOnState dynamics initState

@property
safe : Bool
safe = (globally [0, T - 1] (
          let xLead = transpose trajectory ! 0 in
          let xEgo  = transpose trajectory ! 3 in
          let vEgo  = transpose trajectory ! 4 in
          let dRel  = xLead - xEgo in
          let dSafe = const dDefault [T] + const tGap [T] * vEgo in
          dRel >=. dSafe)) ! 0

@property
comfortable : Bool
comfortable = (globally [0, T - 1] (
                 let gEgo = transpose trajectory ! 5 in
                 const (-comfortMax) [T] <=. gEgo and gEgo <=. const comfortMax [T])) ! 0

@property
respondsToBrake : Bool
respondsToBrake = (globally [0, T - 6] (
                     let xLead = transpose trajectory ! 0 in
                     let xEgo  = transpose trajectory ! 3 in
                     let vEgo  = transpose trajectory ! 4 in
                     let vLead = transpose trajectory ! 1 in
                     let dRel  = xLead - xEgo in
                     let dSafe = const dDefault [T] + const tGap [T] * vEgo in
                     implies (dRel <=. dSafe + const dDefault [T])
                             (finally [0, 5] (vEgo <=. vLead + const epsVGlobal [T])))) ! 0

@property
stabilizes : Bool
stabilizes = (finally [0, T - 6] (
                globally [0, 5] (
                  let xLead = transpose trajectory ! 0 in
                  let xEgo  = transpose trajectory ! 3 in
                  let vEgo  = transpose trajectory ! 4 in
                  let vLead = transpose trajectory ! 1 in
                  let dRel  = xLead - xEgo in
                  let dSafe = const dDefault [T] + const tGap [T] * vEgo in
                  dRel >=. dSafe and vEgo <=. vLead + const epsVGlobal [T]))) ! 0

@property
cruiseUntilFollow : Bool
cruiseUntilFollow = (until [0, T - 1]
                       (let xLead = transpose trajectory ! 0 in
                        let xEgo  = transpose trajectory ! 3 in
                        let vEgo  = transpose trajectory ! 4 in
                        let dRel  = xLead - xEgo in
                        let dSafe = const dDefault [T] + const tGap [T] * vEgo in
                        dRel >=. dSafe)
                       (globally [0, 5] (
                          let vEgo  = transpose trajectory ! 4 in
                          let vLead = transpose trajectory ! 1 in
                          vEgo <=. vLead + const epsVGlobal [T]))) ! 0
