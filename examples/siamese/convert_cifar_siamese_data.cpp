//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_siamese_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <array>
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;


void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}


void convert_dataset(const std::string& input_folder, const std::string& output_folder, const std::string& db_type)
{
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());

  LOG(INFO) << "Writing Training data";
#if 0
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    string batchFileName = input_folder + "/data_batch_"
      + caffe::format_int(fileid+1) + ".bin";
    std::ifstream data_file(batchFileName.c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
    }
  }
#else
    {
        const auto cifar_train_size = kCIFARBatchSize * kCIFARTrainBatches;
        using image_t = std::array<char, kCIFARImageNBytes>;
        std::vector<image_t> all_train_images(cifar_train_size);
        std::vector<int> all_train_labels(cifar_train_size);

        for (int fileid{}; fileid < kCIFARTrainBatches; ++fileid) {
            // Open files
            LOG(INFO) << "Training Batch " << fileid + 1;
            string batchFileName = input_folder + "/data_batch_"
                                   + caffe::format_int(fileid + 1) + ".bin";
            std::ifstream data_file(batchFileName.c_str(),
                                    std::ios::in | std::ios::binary);
            CHECK(data_file) << "Unable to open train file #" << fileid + 1;

            for (int itemid{}; itemid < kCIFARBatchSize; ++itemid) {
                const auto buffer_index = kCIFARBatchSize * fileid + itemid;
                int &label = all_train_labels[buffer_index];
                read_image(&data_file, &label, all_train_images[buffer_index].data());
            }
        }
        std::set<int> labels(all_train_labels.begin(), all_train_labels.end());

        Datum datum;
        datum.set_channels(6);
        datum.set_height(kCIFARSize);
        datum.set_width(kCIFARSize);

        for (int i{}; i < kCIFARTrainBatches * kCIFARBatchSize; ++i) {
            const auto identity_pair = std::rand() % 2;

            const auto random_label_except = [&](int except_lbl) {
                for (;;) {
                    auto candidate = labels.begin();
                    std::advance(candidate, std::rand() % labels.size());
                    if (except_lbl != *candidate) return *candidate;
                }
            };
            const auto random_image_with_label = [&](const int label) {
                for (;;) {
                    const auto index = std::rand() % all_train_images.size();
                    if (all_train_labels[index] == label) {
                        return all_train_images[index];
                    }
                }
            };
            const auto label = identity_pair ? all_train_labels[i] : random_label_except(all_train_labels[i]);
            const auto paired_image = random_image_with_label(label);

            std::vector<char> pixels_buffer(all_train_images[i].begin(), all_train_images[i].end());
            pixels_buffer.insert(pixels_buffer.end(), paired_image.begin(), paired_image.end());

            datum.set_label(identity_pair);
            datum.set_data(pixels_buffer.data(), kCIFARImageNBytes);

            string out;
            CHECK(datum.SerializeToString(&out));
            txn->Put(caffe::format_int(i, 5), out);
        }
    }
#endif
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(), std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
#if 0
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
#else
    {
        using image_t = std::array<char, kCIFARImageNBytes>;
        std::vector<image_t> all_test_images(kCIFARBatchSize);
        std::vector<int> all_test_labels(kCIFARBatchSize);

        for (int itemid{}; itemid < kCIFARBatchSize; ++itemid) {
            read_image(&data_file, &all_test_labels[itemid], all_test_images[itemid].data());
        }

        std::set<int> labels(all_test_labels.begin(), all_test_labels.end());

        Datum datum;
        datum.set_channels(6);
        datum.set_height(kCIFARSize);
        datum.set_width(kCIFARSize);

        for (int i{}; i < kCIFARBatchSize; ++i) {
            const auto identity_pair = std::rand() % 2;

            const auto random_label_except = [&](int except_lbl) {
                for (;;) {
                    auto candidate = labels.begin();
                    std::advance(candidate, std::rand() % labels.size());
                    if (except_lbl != *candidate) return *candidate;
                }
            };
            const auto random_image_with_label = [&](const int label) {
                for (;;) {
                    const auto index = std::rand() % all_test_images.size();
                    if (all_test_labels[index] == label) {
                        return all_test_images[index];
                    }
                }
            };
            const auto label = identity_pair ? all_test_labels[i] : random_label_except(all_test_labels[i]);
            const auto paired_image = random_image_with_label(label);

            std::vector<char> pixels_buffer(all_test_images[i].begin(), all_test_images[i].end());
            pixels_buffer.insert(pixels_buffer.end(), paired_image.begin(), paired_image.end());

            datum.set_label(identity_pair);
            datum.set_data(pixels_buffer.data(), kCIFARImageNBytes);

            string out;
            CHECK(datum.SerializeToString(&out));
            txn->Put(caffe::format_int(i, 5), out);
        }
    }
#endif
  txn->Commit();
  test_db->Close();
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;

  if (argc != 4) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    const std::string input_folder(argv[1]);
    const std::string output_folder(argv[2]);
    const std::string db_type(argv[3]);
    convert_dataset(input_folder, output_folder, db_type);
  }
  return 0;
}
