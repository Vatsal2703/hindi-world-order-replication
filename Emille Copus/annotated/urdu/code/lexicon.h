struct s_entry {

	unichar idno[IDNOLENGTH];

	unichar wordform[WORDLENGTH];

	unichar tag[TAGSMAX][TAGLENGTH];

	unsigned long int freq[TAGSMAX];

	struct s_entry *next_entry;

};
typedef struct s_entry entry;



entry *create_lexicon(const char *filename);

entry *merge_lexicon(entry *lex1, entry *lex2);

entry *tidy_lexicon(entry *lex);

int save_lexicon(entry *head, const char *filename, unsigned int threshold, char probs);

entry *load_lexicon(const char *lex_name);

entry *sort_lexicon(entry *lexicon, int (*sort_entry)(entry *first, entry *second) );

void rewrite_lexnumbers(entry *lexicon);

void free_lexicon(entry *head);

void blank_lexicon(const char *filename, unsigned int lines);

entry *get_entry(void);

entry *find_entry(const unichar *target, entry *head);

void assign_tags(token *dest, entry *source);

int sort_entry_wordform(entry *first, entry *second);
int sort_entry_tag(entry *first, entry *second);


int entry_tagsinc(entry *item, const unichar *thistag);
void entry_addtag(entry *item, const unichar *thistag);