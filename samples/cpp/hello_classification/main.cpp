// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/op/constant.hpp"


#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include "sys/mman.h"

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = TSTRING2STRING(argv[1]);

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        slog::info << "Loading model files: " << model_path << slog::endl;
        core.set_property(ov::enable_mmap(true));

        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        double duration = 0;

        for (int i = 0; i < 50; i++) {
            system("echo 3 > /proc/sys/vm/drop_caches");

            start_time = std::chrono::high_resolution_clock::now();
            std::shared_ptr<ov::Model> model = core.read_model(model_path);

            for (const auto& op : model->get_ordered_ops()) {
                std::shared_ptr<ov::op::v0::Constant> c = ov::as_type_ptr<ov::op::v0::Constant>(op);
                if (c) {
                    auto res = c->get_all_data_elements_bitwise_identical();

                }
            }
            duration += std::chrono::duration_cast<Ms>(std::chrono::high_resolution_clock::now() - start_time).count();
        }
        std::cout << "DURATION: " << duration << std::endl;
        
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
