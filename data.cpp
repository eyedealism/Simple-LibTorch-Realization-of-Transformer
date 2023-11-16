#include <iostream>
#include <torch/torch.h>

// S: Start of decoding input
// E: End of decoding output
// P: Padding, used to fill missing words in sentences shorter than the batch's longest sentence
std::vector<std::vector<std::string>> sentences = {
    // enc_input       dec_input            dec_output
	// make up two stupid examples so that they are not in the training set
    {"ich mochte mein bett essen P", "S i want to eat my bed .", "i want to eat my bed . E"},
    {"ich benutze badewasser zum kochen P", "S i use bath water to cook .", "i use bath water to cook . E"},
};

// Vocabulary, using 0 for padding
// Source vocabulary, in this example, it's the German vocabulary
std::unordered_map<std::string, int> src_vocab = {{"P", 0}, 
                                                  {"ich", 1}, 
                                                  {"mochte", 2}, 
                                                  {"mein", 3}, 
                                                  {"bett", 4}, 
                                                  {"essen", 5},
                                                  {"benutze", 6},
                                                  {"zum", 7},
                                                  {"badewasser", 8},
                                                  {"kochen", 9}};

int src_vocab_size = src_vocab.size(); // 10

// Target vocabulary, in this example, it's the English vocabulary, with additional special symbols
std::unordered_map<std::string, int> tgt_vocab = {{"P", 0}, {"i", 1}, {"want", 2}, {"to", 3}, {"eat", 4}, {"my", 5}, {"bed", 6},
                                                  {"use", 7}, {"bath", 8}, {"water", 9}, {"cook", 10}, {"S", 11}, {"E", 12},
                                                  {".", 13}};

std::unordered_map<int, std::string> idx2word;
for (const auto& kv : tgt_vocab) {
    idx2word[kv.second] = kv.first;
}
int tgt_vocab_size = tgt_vocab.size(); // 14

int src_len = 6; // Maximum sequence length for input enc_input
int tgt_len = 8; // Maximum sequence length for output dec_input/dec_output

// Create tensors for model input
std::tuple<at::Tensor, at::Tensor, at::Tensor> make_data(const std::vector<std::vector<std::string>>& sentences) {
    std::vector<at::Tensor> enc_inputs, dec_inputs, dec_outputs;
    
    for (const auto& sentence : sentences) {
        std::vector<int> enc_input, dec_input, dec_output;
        std::istringstream enc_input_stream(sentence[0]);
        std::istringstream dec_input_stream(sentence[1]);
        std::istringstream dec_output_stream(sentence[2]);
        
        std::string word;
        while (enc_input_stream >> word) {
            enc_input.push_back(src_vocab[word]);
        }
        while (dec_input_stream >> word) {
            dec_input.push_back(tgt_vocab[word]);
        }
        while (dec_output_stream >> word) {
            dec_output.push_back(tgt_vocab[word]);
        }
        
        enc_inputs.push_back(torch::tensor(enc_input, torch::kInt64));
        dec_inputs.push_back(torch::tensor(dec_input, torch::kInt64));
        dec_outputs.push_back(torch::tensor(dec_output, torch::kInt64));
    }
    
    return std::make_tuple(torch::stack(enc_inputs), torch::stack(dec_inputs), torch::stack(dec_outputs));
}

// Create a custom dataset using DataLoader
class MyDataSet : public torch::data::Dataset<MyDataSet> {
public:
    MyDataSet(at::Tensor enc_inputs, at::Tensor dec_inputs, at::Tensor dec_outputs)
        : enc_inputs_(enc_inputs), dec_inputs_(dec_inputs), dec_outputs_(dec_outputs) {}

    torch::data::Example<> get(size_t index) override {
        return {enc_inputs_[index], dec_inputs_[index], dec_outputs_[index]};
    }

    torch::optional<size_t> size() const override {
        return enc_inputs_.size(0);
    }

private:
    at::Tensor enc_inputs_, dec_inputs_, dec_outputs_;
};

int main() {
    // Get the input data
    auto data = make_data(sentences);

    // Create DataLoader
    auto dataset = MyDataSet(std::get<0>(data), std::get<1>(data), std::get<2>(data));
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset,
        torch::data::DataLoaderOptions().batch_size(2).workers(2).shuffle(true)
    );

    // Example usage of the DataLoader
    for (auto& batch : *data_loader) {
        auto enc_inputs = batch.data[0];
        auto dec_inputs = batch.data[1];
        auto dec_outputs = batch.data[2];

        // Process the batch here
    }

    return 0;
}
