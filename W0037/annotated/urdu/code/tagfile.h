#define TAGSMAX 20
#define WORDLENGTH 100
#define TAGLENGTH 10
#define IDNOLENGTH 8
#define USERLENGTH 4

struct s_token {

	unichar s_idno[IDNOLENGTH];

	unichar w_idno[IDNOLENGTH];

	unichar wordform[WORDLENGTH];

	unichar resp[USERLENGTH];

	unichar tag[TAGSMAX][TAGLENGTH];

	unsigned char prob[TAGSMAX];

};

typedef struct s_token token;



token *load_token(FILE *source);

int write_token(token *writeme, FILE *dest);


token *get_token(void);

token *copy_token(token *src);


void clear_tags(token *word);


int tagsinclude(token *word, unichar *searchtag);