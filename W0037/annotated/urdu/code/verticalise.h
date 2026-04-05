
/* constants for return values */

#define SUCCESS 0
#define ERROR_OPEN_INPUT 1
#define ERROR_READ_INPUT 2
#define ERROR_CLOSE_INPUT 3
#define ERROR_OPEN_OUTPUT 4
#define ERROR_WRITE_OUTPUT 5
#define ERROR_CLOSE_OUTPUT 6
#define ERROR_MALLOC 7
#define NONUNICODE 8


int verticalise(const char *input_filename, const char *output_filename, const char *ascii_user);