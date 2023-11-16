#include <torch/torch.h>
#include <iostream>

// Include your Transformer model header
#include "model.h"

int main() {
    // Create a Transformer model and move it to the GPU
    Transformer model;
    model->to(torch::kCUDA);
    model->train();

    // Loss function, ignore classes with index 0 when calculating loss (padding, meaningless)
    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().ignore_index(0));

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(1e-3).momentum(0.99));

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (auto& batch : *data_loader) {
            auto enc_inputs = batch.data[0].to(torch::kCUDA);
            auto dec_inputs = batch.data[1].to(torch::kCUDA);
            auto dec_outputs = batch.data[2].to(torch::kCUDA);

            // Forward pass
            auto outputs = model(enc_inputs, dec_inputs);

            // Calculate loss
            // Reshape dec_outputs to a 1D tensor
            auto loss = criterion(outputs, dec_outputs.view({-1}));

            // Update weights
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            std::cout << "Epoch [" << (epoch + 1) << "/1000], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Save the trained model
    torch::save(model, "MyTransformer_temp.pth");

    return 0;
}
