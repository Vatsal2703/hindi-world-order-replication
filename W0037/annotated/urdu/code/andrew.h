/* my general header file		*/

#ifndef YES
#define YES 1
#endif

#ifndef NO
#define NO 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef NULL
#define NULL 0
#endif

#define invalid_command() {message("Sorry, that is an invalid command.");}

void enter_to_continue(void);
void blank_line(int count);
void clear_screen(void);
void message(const char *string);
int  query(const char *string);
void delay(int length);

void error_file_close(const char *file);
void error_file_open(const char *file);
void error_file_read(const char *file);
void error_file_write(const char *file);
void error_memory_alloc(void);

int  input_yes_no(void);
int  input_integer(void);
char input_one_touch(const char *options);
void input_string(char *target);
void input_hidden_string(char *target);

int  test_file_write(const char *filename);

unsigned char view_file(const char *filename);

int cenputs(const char *string);
int cenfputs(const char *string, FILE *dest);

int strident(const char *string1, const char *string2);
size_t strnident(const char *string1, const char *string2);

char *get_output_textname(const char *input);

void decaps(char *string);