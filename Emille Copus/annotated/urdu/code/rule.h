#define RULESPAN 26


struct s_cond {
	int (*assess)(token *thisword,
				  token *prevwords[RULESPAN],
				  token *nextwords[RULESPAN],
				  short int r,
				  unichar *matchstring);

	short int range;

	unichar matchstring[WORDLENGTH];

	struct s_cond *nextcond;
};
typedef struct s_cond cond;



struct s_rule {

	cond *condlist;

	void (*action)(token *word, unichar *tag);

	unichar tagstring[TAGLENGTH];

	struct s_rule *nextrule;
};
typedef struct s_rule rule;





/* load functions */

rule *load_rulelist(const char *filename);

rule *load_rule(FILE *source);

cond *load_cond(FILE *source);

void load_action(rule *r, FILE *source);




/* apply functions */

int apply_rules_file(rule *rulelist, const char *input_filename, const char *output_filename);

void apply_rule(rule  *thisrule,
				token *thisword,
				token *prevword[RULESPAN],
				token *nextword[RULESPAN]);


void assign_ruleresp(unichar *string, int rulecount);





/* memory allocation functions */

cond *get_cond(void);

rule *get_rule(void);

void free_rulelist(rule *list);




/* the action functions */

void action_assign(token *word, unichar *matchtag);

void action_select(token *word, unichar *matchtag);

void action_delete(token *word, unichar *matchtag);

void action_deletenot(token *word, unichar *matchtag);
