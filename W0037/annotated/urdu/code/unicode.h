/* unicode.h -- prototypes for unicode functions, typedef of unichar type, so on, so forth	*/

/* the defined datatype unichar is 2 bytes and holds a 16-bit unicode character				*/


/* these in-house definitions are used instead of wchar_t because the size of the			*/
/* wchar_t datatype is not necessarily fixed at 2 bytes.									*/
/* the code below however can be engineered to make sure unichar is 2 bytes					*/

/* these lines MUST be adjusted if the functions are to compile on a system with			*/
/* different variable sizes to Win32. unichar MUST be 16 bits and lunichar MUST be 32 bits.	*/

#ifndef ASCII
#define ASCII 8
#endif

#ifndef UNICODE
#define UNICODE 16
#endif


#ifndef UERR
#define UERR 0xffff
#endif

#ifndef RIGHTWAY
#define RIGHTWAY 0xfeff
#endif

typedef unsigned short unichar;


char getbinchar(FILE *source);

unichar fputuc(unichar ch, FILE *dest);

char fputus(const unichar *string, FILE *dest);

unichar fgetuc(FILE *source);

int ungetuc(FILE *fp);

unichar *fgetus(unichar *string, FILE *source);

unichar *fgetnus(unichar *string, size_t n, FILE *source);

size_t ustrlen(const unichar *string);

unichar *ustrcpy(unichar *dest, const unichar *source);

unichar *ustrncpy(unichar *dest, const unichar *source, size_t n);

unichar *ustrdup(const unichar *source);

unichar *ustrcat(unichar *dest, const unichar *source);

unichar *ustrncat(unichar *dest, const unichar *source, size_t n);

long ustrcmp(unichar *str1, unichar *str2);

unichar *ustrchr(unichar *string, const unichar chr);

unichar *ustrrchr(unichar *str, const unichar chr);

size_t ustrcspn(unichar *str1, unichar *str2);
size_t ustrspn(unichar *str1, unichar *str2);

unichar *ustrrev(unichar *string);

unichar *make_unicode(const char *string);

int ustrident(const unichar *str1, const unichar *str2);

size_t ustrnident(const unichar *str1, const unichar *str2);

void ustrimstart(unichar *str, int n);

void ustrimend(unichar *str, int n);

int ucheckdir(FILE *file);

void udecaps(unichar *string);

int whitespace(unichar ch);
int punctuation(unichar ch);
