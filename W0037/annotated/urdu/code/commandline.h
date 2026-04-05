/* general header file for commandline (cl) functions                     */
/* contains alternate, cl_prefixed versions of some functions in andrew.c */

void cl_enter_to_continue(void);
int  cl_query(const char *string);

void cl_error_arguments(int required, int got);
void cl_error_file_close(const char *file);
void cl_error_file_open(const char *file);
void cl_error_file_read(const char *file);
void cl_error_file_write(const char *file);
void cl_error_memory_alloc(void);

int  cl_test_file_write(const char *filename);

// not yet: unsigned char cl_view_file(const char *filename);
