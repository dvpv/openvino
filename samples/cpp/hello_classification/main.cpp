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

#include "openvino/util/mmap_object.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

#include <windows.h>

#define ITERATIONS 5
int b = 0;
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

        std::chrono::time_point<std::chrono::high_resolution_clock> start_time, start_time2;
        double duration = 0;
        double duration2 = 0;

        for (int i = 0; i < ITERATIONS; i++) {
            system("C:\\Users\\vitaliy\\Downloads\\EmptyStandbyList.exe");

            start_time = std::chrono::high_resolution_clock::now();
            
// ################ EXPERIMENT #1 MMAP W/ Prefetch ################
            // auto mapped_memory = ov::load_mmap_object(model_path);
            // WIN32_MEMORY_RANGE_ENTRY memoryRange;
            // memoryRange.VirtualAddress = mapped_memory->data();
            // memoryRange.NumberOfBytes = mapped_memory->size();

            // HANDLE hProcess = GetCurrentProcess();
            // PrefetchVirtualMemory(hProcess, 1, &memoryRange, 0);

// ################ EXPERIMENT #2 MMAP W/O Prefetch ################
            // auto mapped_memory = ov::load_mmap_object(model_path);
            // auto s = mapped_memory->size();
            // auto p = mapped_memory->data();
            // std::vector<char> a;
            // a.reserve(s);
            // start_time2 = std::chrono::high_resolution_clock::now();
            // for (int i = 0; i < s; i++)
            //     a[i] = p[i];
            // std::cout << a[i] << std::endl;
            // std::cout << "SIZE: " << s << std::endl;

// ################ EXPERIMENT #3 READ ################
            // std::ifstream bin_stream;
            // bin_stream.open(model_path.c_str(), std::ios::binary);

            // bin_stream.seekg(0, std::ios::end);
            // size_t file_size = bin_stream.tellg();
            // bin_stream.seekg(0, std::ios::beg);

            // auto aligned_weights_buffer = std::make_shared<ov::AlignedBuffer>(file_size);
            // bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
            // bin_stream.close();

            // auto s = aligned_weights_buffer->size();
            // auto p = aligned_weights_buffer->get_ptr<char>();
            // std::vector<char> a;
            // a.reserve(s);
            // start_time2 = std::chrono::high_resolution_clock::now();
            // for (int i = 0; i < s; i++)
            //     a[i] = p[i];
            // std::cout << a[i] << std::endl;
            // std::cout << "SIZE: " << aligned_weights_buffer->size() << std::endl;

// ################ EXPERIMENT #4 ITERATE OVER CONSTANTS (get_all_data_elements_bitwise_identical() touches each element) ################
            std::stringstream ss;
            core.set_property(ov::enable_mmap(true));
            std::shared_ptr<ov::Model> model = core.read_model(model_path);

            start_time2 = std::chrono::high_resolution_clock::now();
            std::vector<bool> a(ITERATIONS);
            int count = 0;
            for (const auto& op : model->get_ordered_ops()) {
                std::shared_ptr<ov::op::v0::Constant> c = ov::as_type_ptr<ov::op::v0::Constant>(op);
                if (c) {
                    b += c->get_all_data_elements_bitwise_identical();
                }
            }
            a[i] = b;
            ss << a[i];

// ################ EXPERIMENT #5 LLAMA INFERENCE (ALLOCATES MEMORY ON INFERENCE STEP) ################
            auto c_m = core.compile_model(model, "CPU");
            auto req = c_m.create_infer_request();
            req.set_tensor("input_ids", ov::Tensor(ov::element::i64, ov::Shape{1, 100}));
            req.set_tensor("attention_mask", ov::Tensor(ov::element::i64, ov::Shape{1, 100}));
            req.set_tensor("position_ids", ov::Tensor(ov::element::i64, ov::Shape{1, 100}));
            req.set_tensor("beam_idx", ov::Tensor(ov::element::i32, ov::Shape{1}));
            req.infer();
            std::cout << "Inference finished" << std::endl;

            duration += std::chrono::duration_cast<Ms>(std::chrono::high_resolution_clock::now() - start_time).count();
            duration2 += std::chrono::duration_cast<Ms>(std::chrono::high_resolution_clock::now() - start_time2).count();
        }
        std::cout << "DURATION millisecs: " << duration << std::endl;
        std::cout << "ITERATIONS millisecs: " << duration2 << std::endl;
        
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
