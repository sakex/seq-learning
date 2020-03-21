//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_ECOVTYPES_H
#define SEQ_LEARNING_ECOVTYPES_H

#include <exception>
#include <string>

namespace activegp {
    enum eCovTypes {
        gaussian,
        matern_3_2,
        matern_5_2
    };

    struct InvalidCovType : std::exception {
        char const *msg;

        explicit InvalidCovType(char const *_msg) : msg(_msg) {
        }

        virtual const char *what() const noexcept {
            char const beginning[] = "Invalid cov type: '";
            char const ending[] = "'\nExpected: Gaussian/Mattern3_2/Mattern5_2";
            char *out = (char *) malloc(std::strlen(beginning) + std::strlen(msg) + std::strlen(ending) + 1);
            std::strcpy(out, beginning);
            std::strcat(out, msg);
            std::strcat(out, ending);
            return out;
        }
    };
}
#endif //SEQ_LEARNING_ECOVTYPES_H
