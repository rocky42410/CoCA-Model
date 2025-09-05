#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <iomanip>

namespace fs = std::filesystem;

class CSVPartitioner {
private:
    std::string input_file;
    int num_partitions;
    std::string output_dir;
    std::string header;
    std::vector<std::string> data_rows;
    
    std::string get_base_filename() const {
        fs::path p(input_file);
        return p.stem().string();
    }
    
    std::string get_extension() const {
        fs::path p(input_file);
        return p.extension().string();
    }
    
    std::string get_output_filename(int partition_idx) const {
        std::ostringstream oss;
        oss << get_base_filename() 
            << "_part_" 
            << std::setfill('0') << std::setw(std::to_string(num_partitions).length()) 
            << partition_idx + 1
            << get_extension();
        
        // Combine with output directory
        fs::path output_path = fs::path(output_dir) / oss.str();
        return output_path.string();
    }
    
public:
    CSVPartitioner(const std::string& file, int partitions, const std::string& out_dir) 
        : input_file(file), num_partitions(partitions), output_dir(out_dir) {
        
        if (num_partitions <= 0) {
            throw std::invalid_argument("Number of partitions must be positive");
        }
        
        // Create output directory if it doesn't exist
        if (!fs::exists(output_dir)) {
            std::cout << "Creating output directory: " << output_dir << std::endl;
            if (!fs::create_directories(output_dir)) {
                throw std::runtime_error("Failed to create output directory: " + output_dir);
            }
        } else if (!fs::is_directory(output_dir)) {
            throw std::runtime_error("Output path exists but is not a directory: " + output_dir);
        }
    }
    
    void read_csv() {
        std::ifstream file(input_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open input file: " + input_file);
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (first_line) {
                header = line;
                first_line = false;
            } else if (!line.empty()) {  // Skip empty lines
                data_rows.push_back(line);
            }
        }
        
        file.close();
        
        if (header.empty()) {
            throw std::runtime_error("CSV file appears to be empty");
        }
        
        std::cout << "Read " << data_rows.size() << " data rows from " << input_file << std::endl;
    }
    
    void partition() {
        if (data_rows.empty()) {
            throw std::runtime_error("No data rows to partition");
        }
        
        // Calculate partition sizes
        size_t total_rows = data_rows.size();
        size_t base_size = total_rows / num_partitions;
        size_t remainder = total_rows % num_partitions;
        
        // If we have fewer rows than partitions, adjust partition count
        if (total_rows < static_cast<size_t>(num_partitions)) {
            std::cout << "Warning: Fewer rows (" << total_rows 
                      << ") than requested partitions (" << num_partitions 
                      << "). Creating " << total_rows << " partitions instead." << std::endl;
            num_partitions = total_rows;
            base_size = 1;
            remainder = 0;
        }
        
        size_t current_row = 0;
        
        for (int i = 0; i < num_partitions; ++i) {
            // Calculate size for this partition
            size_t partition_size = base_size + (i < static_cast<int>(remainder) ? 1 : 0);
            
            if (partition_size == 0) {
                continue;  // Skip empty partitions (shouldn't happen with the adjustment above)
            }
            
            std::string output_filename = get_output_filename(i);
            std::ofstream out_file(output_filename);
            
            if (!out_file.is_open()) {
                throw std::runtime_error("Failed to create output file: " + output_filename);
            }
            
            // Write header to each partition
            out_file << header << std::endl;
            
            // Write data rows for this partition
            size_t end_row = std::min(current_row + partition_size, total_rows);
            for (size_t j = current_row; j < end_row; ++j) {
                out_file << data_rows[j] << std::endl;
            }
            
            out_file.close();
            
            std::cout << "Created " << output_filename 
                      << " with " << (end_row - current_row) << " data rows" << std::endl;
            
            current_row = end_row;
        }
        
        std::cout << "\nPartitioning complete. Created " << num_partitions 
                  << " files in " << output_dir << std::endl;
    }
    
    void verify_partitions() {
        std::cout << "\nVerifying partitions can be concatenated to recreate original..." << std::endl;
        
        size_t total_verified_rows = 0;
        std::string combined_header;
        
        for (int i = 0; i < num_partitions; ++i) {
            std::string filename = get_output_filename(i);
            std::ifstream file(filename);
            
            if (!file.is_open()) {
                std::cerr << "Warning: Could not open " << filename << " for verification" << std::endl;
                continue;
            }
            
            std::string line;
            bool first_line = true;
            size_t file_rows = 0;
            
            while (std::getline(file, line)) {
                if (first_line) {
                    if (combined_header.empty()) {
                        combined_header = line;
                    } else if (combined_header != line) {
                        std::cerr << "Warning: Header mismatch in " << filename << std::endl;
                    }
                    first_line = false;
                } else if (!line.empty()) {
                    file_rows++;
                    total_verified_rows++;
                }
            }
            
            file.close();
            
            // Extract just the filename for display
            fs::path p(filename);
            std::cout << "  " << p.filename().string() << ": " << file_rows << " data rows" << std::endl;
        }
        
        if (total_verified_rows == data_rows.size()) {
            std::cout << "✓ Verification successful: " << total_verified_rows 
                      << " total rows across all partitions" << std::endl;
        } else {
            std::cerr << "✗ Verification failed: Expected " << data_rows.size() 
                      << " rows but found " << total_verified_rows << std::endl;
        }
    }
    
    void print_concatenation_command() const {
        std::cout << "\nTo concatenate back to original (Linux/Mac):" << std::endl;
        
        // Build path pattern for the partition files
        fs::path pattern = fs::path(output_dir) / (get_base_filename() + "_part_*.csv");
        fs::path first_file = fs::path(output_dir) / (get_base_filename() + "_part_1.csv");
        fs::path output_file = fs::path(output_dir) / "combined.csv";
        
        std::cout << "  head -n 1 \"" << first_file.string() << "\" > \"" 
                  << output_file.string() << "\"" << std::endl;
        std::cout << "  for f in \"" << pattern.string() << "\"; do tail -n +2 \"$f\" >> \"" 
                  << output_file.string() << "\"; done" << std::endl;
        
        std::cout << "\nOr if running from a different directory:" << std::endl;
        std::cout << "  cd \"" << output_dir << "\"" << std::endl;
        std::cout << "  head -n 1 " << get_base_filename() << "_part_1.csv > combined.csv" << std::endl;
        std::cout << "  for f in " << get_base_filename() << "_part_*.csv; do tail -n +2 \"$f\" >> combined.csv; done" << std::endl;
    }
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <csv_file> <num_partitions> <output_directory>" << std::endl;
    std::cerr << "\nDescription:" << std::endl;
    std::cerr << "  Divides a CSV file into the specified number of partitions chronologically." << std::endl;
    std::cerr << "  Each partition maintains the original header and can be concatenated" << std::endl;
    std::cerr << "  to recreate the original file (excluding headers from partitions 2+)." << std::endl;
    std::cerr << "\nArguments:" << std::endl;
    std::cerr << "  csv_file          - Path to the input CSV file" << std::endl;
    std::cerr << "  num_partitions    - Number of partitions to create (positive integer)" << std::endl;
    std::cerr << "  output_directory  - Directory where partition files will be saved" << std::endl;
    std::cerr << "                      (will be created if it doesn't exist)" << std::endl;
    std::cerr << "\nExample:" << std::endl;
    std::cerr << "  " << program_name << " data.csv 4 ./partitions" << std::endl;
    std::cerr << "  Creates: ./partitions/data_part_1.csv through data_part_4.csv" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string csv_file = argv[1];
    int num_partitions;
    std::string output_dir = argv[3];
    
    try {
        num_partitions = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid partition number '" << argv[2] << "'" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        // Check if input file exists
        if (!fs::exists(csv_file)) {
            throw std::runtime_error("Input file does not exist: " + csv_file);
        }
        
        CSVPartitioner partitioner(csv_file, num_partitions, output_dir);
        
        // Read the CSV file
        partitioner.read_csv();
        
        // Perform partitioning
        partitioner.partition();
        
        // Verify the partitions
        partitioner.verify_partitions();
        
        // Print concatenation command for user reference
        partitioner.print_concatenation_command();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}