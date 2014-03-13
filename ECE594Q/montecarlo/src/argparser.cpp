#include "argparser.h"

char* ArgParser::GetCmdOpt(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool ArgParser::CmdOptExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}
