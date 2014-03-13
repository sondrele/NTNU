#ifndef _ARGPARSER_H_
#define _ARGPARSER_H_

#include <algorithm>
#include <string>

class ArgParser {
public:
    static char * GetCmdOpt(char **, char **, const std::string &);
    static bool CmdOptExists(char **, char **, const std::string &);
};

#endif // _ARGPARSER_H_