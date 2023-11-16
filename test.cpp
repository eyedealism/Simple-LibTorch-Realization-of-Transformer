#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "model.h"

// Define a function for greedy decoding
std::vector<long> greedy_decoder(Transformer& model, const torch::Tensor& enc_input, int start_symbol) {
    // Get encoder outputs
    torch::Tensor enc_outputs = model.encoder(enc_input);
    
    // Initialize decoder input with start symbol
    torch::Tensor dec_input = torch::zeros({1, 0}).to(enc_input.device());
    int next_symbol = start_symbol;
    bool flag = true;
    std::vector<long> decoded_sequence;
    
    while (flag) {
        // Append the next_symbol to the decoder input
        dec_input = torch::cat({dec_input.detach(), torch::tensor({{next_symbol}}, torch::dtype(enc_input.dtype()).device(enc_input.device()))}, -1);
        
        // Generate decoder outputs
        torch::Tensor dec_outputs = model.decoder(dec_input, enc_input, enc_outputs);
        torch::Tensor projected = model.projection(dec_outputs);
        
        // Get the most probable next symbol
        torch::Tensor prob = projected.squeeze(0).argmax(-1);
        next_symbol = prob.item<long>();
        decoded_sequence.push_back(next_symbol);
        
        // Check if the next symbol is the end symbol
        if (next_symbol == tgt_vocab['.']) {
            flag = false;
        }
        std::cout << next_symbol << std::endl;
    }
    
    return decoded_sequence;
}

int main() {
    // Load the trained model
    Transformer model;
    torch::load(model, "MyTransformer_temp.pth");
    model->eval();
    model->to(torch::kCUDA);
    
    // Perform greedy decoding for a batch of input sequences
    torch::NoGradGuard no_grad;  // Disable gradient tracking
    
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset,
        torch::data::DataLoaderOptions().batch_size(2).workers(2).shuffle(true)
    );
    
    for (auto& batch : *data_loader) {
        auto enc_inputs = batch.data[0].to(torch::kCUDA);
        
        for (int i = 0; i < enc_inputs.size(0); ++i) {
            std::vector<long> greedy_dec_input = greedy_decoder(model, enc_inputs[i].view({1, -1}), tgt_vocab['S']);
            torch::Tensor predict = model(enc_inputs[i].view({1, -1}), greedy_dec_input);  // predict: [batch_size * tgt_len, tgt_vocab_size]
            predict = predict.argmax(-1).view({-1}).to(torch::kCPU);
            
            // Print the input sequence and the predicted output
            std::cout << enc_inputs[i] << " -> ";
            for (long n : predict) {
                std::cout << idx2word[n.item()] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    return 0;
}
