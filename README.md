# ApproxMCPB: Approximate Model Counter for PB Constraints

[![CP](https://img.shields.io/badge/CP-2021-blue.svg)](https://drops.dagstuhl.de/opus/volltexte/2021/15349/)
[![Dataset](https://img.shields.io/badge/paper-Dataset-yellow.svg)](https://doi.org/10.5281/zenodo.5526834)
[![Underlying Solver](https://img.shields.io/badge/solver-LinPB-red.svg)](https://github.com/meelgroup/linpb)

Approximate Model Counter for linear pseudo boolean and xor constraints, built on top of the PB-XOR solver [LinPB](https://github.com/meelgroup/linpb) and approximate model counter [ApproxMC](https://github.com/meelgroup/approxmc). 
The counter is expected to perform well on high-level PB constraints, instead of encoding PB constraints into CNF.

## Input format
   - Pseudo Boolean constraints: [OPB format](InputFormats.md)
   - XOR constraints: `* xor <variable>+ 0|1`
   
### Example input file

```
* #variable= 4 #constraint= 2
*
* ind 1 2 3 4 0
*
* xor x1 x2 1
* xor x3 x4 0
* 
+1 x1 +2 x2 >= 1;
+1 x1 +2 x3 -3 x4 = -1;
```

Note: We recommend to encode short xors into pb constraints, while for larger xors please use XOR constraints directly.

### Sampling Set Format
If sampling set is {1,2,3,4}, then add the following line to input file:
```
* ind 1 2 3 4 0
```
You can specify sampling set in multiline, and make sure each line ends with '0'. For example, an equivalent input is:
```
* ind 1 2 0
* ind 3 4 0
```

## How to Build
To build on Ubuntu, you will need the following:
```
sudo apt-get install build-essential cmake libgmp-dev
sudo apt-get install zlib1g-dev libboost-program-options-dev libm4ri-dev
```

CentOS:
```
yum install gcc gcc-c++ kernel-devel cmake
yum install zlib-devel zlib-static boost-devel
```
You have to install m4ri manually on CentOS.

Then, build ApproxMCPB:
```
git clone --recurse-submodules https://github.com/meelgroup/approxmcpb.git
cd approxmcpb
mkdir build && cd build
cmake ..
make
sudo make install
```

### Static Compilation: 
You have to install static library at frist. Let's take gmp as an example:
```
yum install gmp-static
```
Then add an extra cmake option `-DSTATICCOMPILE=ON`:
```
cmake -DSTATICCOMPILE=ON ..
```

### Keep LinPB Up-to-date
To fetch the latest LinPB, run:
```
git pull
git submodule update --init
```

### SoPlex

RoundingSat Underlying LinPB supports an integration with the LP solver SoPlex to improve its search routine.

For this, first download SoPlex at [here](https://soplex.zib.de/download.php?fname=soplex-5.0.1.tgz) and place the downloaded file in the root directory of LinPB, i.e. ./src/linpb/ .
Next, follow the above build process, but configure with the cmake option `-Dsoplex=ON`:

```
cmake -Dsoplex=ON ..
```

## How to Use
Run the following command:
```
$ approxmcpb /path/to/file
```
Find more arguments: 
```
$ approxmcpb --help
```

### Potential Bug

You may encounter a failed assertion if using SoPlex:

`spxdevexpr.hpp:695: soplex::SPxId soplex::SPxDevexPR<R>::selectEnterHyperCoDim(R&, R) [with R = double]: 
Assertion 'bestPricesCo.size() == 0' failed.`

The bug will be fixed in the upcoming bugfix release of SoPlex, clarified by SoPlex team. Before that, you can simply comment out this assertion.

Integration with SoPlex is under test. We recommend to remove it if you encountered other errors with SoPlex.

## References

### ApproxMCPB

**[YM21]** Jiong Yang, Kuldeep S. Meel. Engineering an Efficient PB-XOR Solver. *CP 2021*

### ApproxMC

**[CMV16]** Supratik Chakraborty, Kuldeep S. Meel, Moshe Y. Vardi. A Scalable Approximate Model Counter. *CP 2013*

**[SGM20]** Soos M., Gocht S., Meel K.S. Tinted, Detached, and Lazy CNF-XOR Solving and Its Applications to Counting and Sampling. *CAV 2020*
