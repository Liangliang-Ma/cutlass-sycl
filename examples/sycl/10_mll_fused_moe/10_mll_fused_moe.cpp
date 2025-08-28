#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

#include <cfloat>

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

using ElementAccumulator = float;     // <- data type of accumulator
using ElementComputeEpilogue = float; // <- data type of epilogue operations
using ElementOutput = float;          // <- data type of elements in output matrix D

// Command line options parsing
struct Options {

  bool error = false;
  bool help = false;

  float alpha, beta;
  int iterations;
  int m, n, k, groups;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  Options() : error(false), help(false), alpha(FLT_MAX), beta(FLT_MAX), iterations(100),
              m(5120), n(4096), k(4096), groups(2) {
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("groups", groups, 2);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta",  beta,  0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);

    assert(groups > 0);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "BMG Grouped GEMM\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                   Sets the N extent of the GEMM for all groups\n"
      << "  --k=<int>                   Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "09_bmg_grouped_gemm_fp8" << " --m=5120 --n=4096 --k=4096 --groups=5 --alpha=2.5 --beta=0.5 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////



int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }
  
  printf("hello cutlass\n");
  // launcher<cutlass::float_e5m2_t>(options);
  // launcher<cutlass::float_e4m3_t>(options);
  return 0;
}


