#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "andrew.h"
#include "commandline.h"
#include "unicode.h"
#include "tagfile.h"
#include "rule.h"
#include "assess.h"

/* load functions */

/* load a rule list in its entirety  */
rule *load_rulelist(const char *filename)
{
	rule *list;
	rule *r;
	rule *current;

	FILE *source;


	/* open the source file and check for (then discard) directionality character */
	if (!(source = fopen(filename, "rb")))
	{
		cl_error_file_open(filename);
		fcloseall();
		return NULL;
	}
	if (!( ucheckdir(source) ))
	{
		puts("Unicode file not recognised. Rulelist could not load.");
		fcloseall();
		return NULL;
	}



	list = NULL;

	while (1)
	{
		/* run load_rule */
		r = load_rule(source);

		/* if it's NULL, we've got to end-of-file or a badly formed line, therefore break */
		if ( ! r )
			break;

		/* otherwise add the new rule to the list */
		if (!list)
		{
			list = r;
			current = list;
		}
		else
		{
			current->nextrule = r;
			current = current->nextrule;
		}
	}

	if (fclose(source))
	{
		cl_error_file_close(filename);
		fcloseall();
	}

	/* return list, which will be NULL if no rule was successfully acquired */
	return list;
}



/* load a rule from file - returns NULL if this file contains no more valid rules */
rule *load_rule(FILE *source)
{
	unichar junk[500];

	rule *r;

	cond *c;

	cond *c_latest;


	r = get_rule();

	while (1)
	{
		/* load two characters  - in case of end-of-file, break */
		if ( (junk[0] = fgetuc(source)) == UERR)
			break;
		if ( (junk[1] = fgetuc(source)) == UERR)
			break;

		
		/* if it's a line break, i.e. empty line, continue */
		if ( (junk[0] == 0x000d) && junk[1] == 0x000a )
			continue;


		/* if it's "c ", load a condition to the end of the condlist... */
		if ( (junk[0] == 0x0063) && junk[1] == 0x0020 )
		{
			c = load_cond(source);

			/* in case of end-of-file or badly-formed line, break */
			if ( ! c )
				break;
			
			/* otherwise, add it to the cond list, and continue (i.e. go to next line) */
			if ( ! r->condlist )
			{
				r->condlist = c;
				c_latest = r->condlist;
			}
			else
			{
				c_latest->nextcond = c;
				c_latest = c_latest->nextcond;
			}
			continue;
		}


		/* if it's "a ", load_action and break */
		if ( (junk[0] == 0x0061) && junk[1] == 0x0020 )
		{
			load_action(r, source);
			break;
		}



		/* anything else, load a line to junk and continue, breaking if EOF is reached */
		if ( !fgetus(junk, source) )
			break;
	}


	/* in case of break, free(r) and r = NULL if it's an incomplete action */
	/* (i.e. end_of_file reached by one of the sub_functions) */
	if ( ! r->action )
	{
		free(r);
		r = NULL;
	}

	return r;
}



/* load_cond returns NULL for end-of-file or a badly formed line */
cond *load_cond(FILE *source)
{
	int i;
	char c_junk[50];

	static unichar str_ifthiswordis[50];
	static unichar str_ifthistagis[50];
	static unichar str_ifthistaginc[50];
	static unichar str_ifprevwordis[50];
	static unichar str_ifprevtagis[50];
	static unichar str_ifprevtaginc[50];
	static unichar str_ifnextwordis[50];
	static unichar str_ifnexttagis[50];
	static unichar str_ifnexttaginc[50];
	static unichar str_ifthiswordisnot[50];
	static unichar str_ifthistagisnot[50];
	static unichar str_ifthistagincnot[50];
	static unichar str_ifprevwordisnot[50];
	static unichar str_ifprevtagisnot[50];
	static unichar str_ifprevtagincnot[50];
	static unichar str_ifnextwordisnot[50];
	static unichar str_ifnexttagisnot[50];
	static unichar str_ifnexttagincnot[50];
	static char herebefore = NO;

	unichar buffer[500];

	cond *c;



	/* initialise compare strings on first call */
	if (!herebefore)
	{
		strcpy(c_junk, "ifthiswordis");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifthiswordis[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifthistagis");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifthistagis[i]  = c_junk[i]) ) break; }
		strcpy(c_junk, "ifthistaginc");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifthistaginc[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifprevwordis");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifprevwordis[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifprevtagis");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifprevtagis[i]  = c_junk[i]) ) break; }
		strcpy(c_junk, "ifprevtaginc");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifprevtaginc[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifnextwordis");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifnextwordis[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifnexttagis");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifnexttagis[i]  = c_junk[i]) ) break; }
		strcpy(c_junk, "ifnexttaginc");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifnexttaginc[i] = c_junk[i]) ) break; }

		strcpy(c_junk, "ifthiswordisnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifthiswordisnot[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifthistagisnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifthistagisnot[i]  = c_junk[i]) ) break; }
		strcpy(c_junk, "ifthistagincnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifthistagincnot[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifprevwordisnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifprevwordisnot[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifprevtagisnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifprevtagisnot[i]  = c_junk[i]) ) break; }
		strcpy(c_junk, "ifprevtagincnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifprevtagincnot[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifnextwordisnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifnextwordisnot[i] = c_junk[i]) ) break; }
		strcpy(c_junk, "ifnexttagisnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifnexttagisnot[i]  = c_junk[i]) ) break; }
		strcpy(c_junk, "ifnexttagincnot");
		for ( i = 0 ; 1 ; i++ ) { if (!(str_ifnexttagincnot[i] = c_junk[i]) ) break; }

		herebefore = YES;
	}


	c = get_cond();



	/* load the cond-type (all chars to next space) */
	for ( i = 0 ; i < 500 ; i++ )
	{
		if ( (buffer[i] = fgetuc(source)) == UERR)
		{
			/* shouldn't be an EOF here! */
			free(c);
			return NULL;
		}
		if ( buffer[i] == 0x0020 )
		{
			buffer[i] = 0;
			break;
		}
	}
	/* assign a function pointer */
	udecaps(buffer);

	if (ustrident(buffer, str_ifthiswordis))
		c->assess = assess_ifthiswordis;

	if (ustrident(buffer, str_ifthistagis))
		c->assess = assess_ifthistagis;

	if (ustrident(buffer, str_ifthistaginc))
		c->assess = assess_ifthistaginc;

	if (ustrident(buffer, str_ifprevwordis))
		c->assess = assess_ifprevwordis;

	if (ustrident(buffer, str_ifprevtagis))
		c->assess = assess_ifprevtagis;

	if (ustrident(buffer, str_ifprevtaginc))
		c->assess = assess_ifprevtaginc;

	if (ustrident(buffer, str_ifnextwordis))
		c->assess = assess_ifnextwordis;

	if (ustrident(buffer, str_ifnexttagis))
		c->assess = assess_ifnexttagis;

	if (ustrident(buffer, str_ifnexttaginc))
		c->assess = assess_ifnexttaginc;

	if (ustrident(buffer, str_ifthiswordisnot))
		c->assess = assess_ifthiswordisnot;

	if (ustrident(buffer, str_ifthistagisnot))
		c->assess = assess_ifthistagisnot;

	if (ustrident(buffer, str_ifthistagincnot))
		c->assess = assess_ifthistagincnot;

	if (ustrident(buffer, str_ifprevwordisnot))
		c->assess = assess_ifprevwordisnot;

	if (ustrident(buffer, str_ifprevtagisnot))
		c->assess = assess_ifprevtagisnot;

	if (ustrident(buffer, str_ifprevtagincnot))
		c->assess = assess_ifprevtagincnot;

	if (ustrident(buffer, str_ifnextwordisnot))
		c->assess = assess_ifnextwordisnot;

	if (ustrident(buffer, str_ifnexttagisnot))
		c->assess = assess_ifnexttagisnot;

	if (ustrident(buffer, str_ifnexttagincnot))
		c->assess = assess_ifnexttagincnot;

	if (c->assess == NULL)
	{
		/* badly formed line */
		free(c);
		return NULL;
	}



	if (c->assess != assess_ifthiswordis &&
		c->assess != assess_ifthistagis  &&
		c->assess != assess_ifthistaginc &&
		c->assess != assess_ifthiswordisnot &&
		c->assess != assess_ifthistagisnot  &&
		c->assess != assess_ifthistagincnot )
	{
		/* load the range (to next space) if it's not a "this" assessor */
		for ( i = 0 ; i < 500 ; i++ )
		{
			if ( (buffer[i] = fgetuc(source)) == UERR)
			{
				/* shouldn't be an EOF here! */
				free(c);
				return NULL;
			}
			if ( buffer[i] == 0x0020 )
			{
				buffer[i] = 0;
				break;
			}
			if ( buffer[i] > 0x0039 || buffer[i] < 0x0030 )
			{
				/* i.e. if the char is not a number */
				free(c);
				return NULL;
			}
		}

		/* convert to a number, assign to c->range */
		if ( (i = ustrlen(buffer)) > 2 )
		{
			/* number is too big ! */
			free(c);
			return NULL;
		}
		if ( i == 2 )
			c->range = (buffer[1] - 0x0030) * 10 ;
		else
			c->range = 0;

		c->range += (buffer[0] - 0x0030);

		if (c->range >= RULESPAN)
		{
			/* number is too big ! */
			free(c);
			return NULL;
		}

	} /* if it's a "this" assessor, just carry on... */


	/* load the matchstring (all chars to next whitespace) */
	for ( i = 0 ; i < 500 ; i++ )
	{
		if ( (buffer[i] = fgetuc(source)) == UERR)
		{
			buffer[i] = 0;
			break;
			/* end of file reached, but AFTER the condition was completed, so don't wipe */
		}
		if ( whitespace(buffer[i]) )
		{
			buffer[i] = 0;
			break;
		}
	}
	/* if buffer is empty, free c and return NULL, else move it to matchstring */
	if ( ! buffer[0] )
	{
		free(c);
		return NULL;
	}
	else
		ustrcpy(c->matchstring, buffer);


	/* scroll file to end of line (while char loaded != 0x0a) */
	buffer[0] = 0;
	while ( buffer[0] != 0x000a )
	{
		if ( (buffer[0] = fgetuc(source)) == UERR)
		{
			/* end of file reached, whihch shouldn't happen in a cond; therefore wipe */
			free(c);
			return NULL;
		}
	}

	return c;
}



/* if entire action is not retrieved, r->action = NULL and nothing in tagstring */
void load_action(rule *r, FILE *source)
{
	int i;

	unichar buffer[500];

	unichar str_assign[] = { 0x0061, 0x0073, 0x0073, 0x0069, 0x0067, 0x006e, 0x0000 };
	unichar str_select[] = { 0x0073, 0x0065, 0x006c, 0x0065, 0x0063, 0x0074, 0x0000 };
	unichar str_delete[] = { 0x0064, 0x0065, 0x006c, 0x0065, 0x0074, 0x0065, 0x0000 };
	unichar str_deletenot[]
		 = { 0x0064, 0x0065, 0x006c, 0x0065, 0x0074, 0x0065, 0x006e, 0x006f, 0x0074, 0x0000 };

	r->action = NULL;


	/* load the name of the action (all chars to next space) */
	for ( i = 0 ; i < 500 ; i++ )
	{
		if ( (buffer[i] = fgetuc(source)) == UERR)
		{
			buffer[i] = 0;
			break;
			/* if the action is complete, then the EOF will be caught below */
		}
		if ( buffer[i] == 0x0020 )
		{
			buffer[i] = 0;
			break;
		}
	}

	/* assign a pointer */
	udecaps(buffer);

	if (ustrident(buffer, str_assign))
		r->action = action_assign;
	
	if (ustrident(buffer, str_select))
		r->action = action_select;

	if (ustrident(buffer, str_delete))
		r->action = action_delete;

	if (ustrident(buffer, str_deletenot))
		r->action = action_deletenot;

	/* if no pointer assigned, then no valid action has been detected; therefore return */
	if ( r->action == NULL )
		return;


	/* load the target tag (all chars to next whitespace) */
	for ( i = 0 ; i < 500 ; i++ )
	{
		if ( (buffer[i] = fgetuc(source)) == UERR)
		{
			buffer[i] = 0;
			break;
			/* end of file reached, but AFTER the rule was completed, so don't wipe */
		}
		if ( whitespace(buffer[i]) )
		{
			buffer[i] = 0;
			break;
		}
	}

	/* if this string is empty, nullify r->action and return; otherwise, copy it to the rule */
	if ( ! buffer[0] )
	{
		r->action = NULL;
		r->tagstring[0] = 0x0000;
		return;
	}
	else
		ustrcpy(r->tagstring, buffer);

	/* scroll file to end of line (while char != 0x0a) */
	buffer[0] = 0;
	while ( buffer[0] != 0x000a )
	{
		if ( (buffer[0] = fgetuc(source)) == UERR)
			return;
		/* end of file reached, but AFTER the rule was completed, so don't wipe */
	}
}



/* application functions */

/* returns 0 for all OK, 1 for "there was an error!" */
int apply_rules_file(rule *rulelist, const char *input_filename, const char *output_filename)
{
	int i;
	int rulecount;

	unichar tagbuffer[TAGSMAX][TAGLENGTH];

	FILE *source;
	FILE *dest;

	token *thisword;

	token *prev[RULESPAN];

	token *next[RULESPAN];

	rule *thisrule;


	/* open the source file and check for (then discard) directionality character */
	if (!(source = fopen(input_filename, "rb")))
	{
		cl_error_file_open(input_filename);
		fcloseall();
		return 1;
	}
	if (!( ucheckdir(source) ))
	{
		fputs("Specified source file not recognised as Unicode!", stderr);
		fcloseall();
		return 1;
	}

	/* open file to write, insert directionality character */
	if( !(dest = fopen(output_filename, "wb")) )
	{
		cl_error_file_open(output_filename);
		fcloseall();
		return 1;
	}
	if ( fputuc( RIGHTWAY , dest) == UERR )
	{
		cl_error_file_write(output_filename);
		fcloseall();
		return 1;
	}


	/* initialise all token pointers in prev to NULL */
	for ( i = 0 ; i < RULESPAN ; i++)
		prev[i] = NULL;

	/* load an initial set of tokens and assign their pointers to next */
	for ( i = 0 ; i < RULESPAN ; i++ )
		next[i] = load_token(source);

	/* assign token in next[0] to prev[0] and thisword too */
	prev[0] = (thisword = next[0]);


	while ( thisword != NULL )
	{
		/* apply all the rules to thisword */
		for ( thisrule = rulelist, rulecount = 1 ;
			  thisrule != NULL ; 
			  thisrule = thisrule->nextrule, rulecount++ )
		{
			/* copy tags to a buffer */
			for ( i = 0 ; i < TAGSMAX ; i++ )
				ustrcpy(tagbuffer[i], thisword->tag[i]);


			apply_rule(thisrule, thisword, prev, next);
			
			
			/* add a resptag if there was a change */
			for ( i = 0 ; i < TAGSMAX ; i++ )
			{
				if ( ! ustrident(tagbuffer[i], thisword->tag[i]) )
				{
					/* add a resptag using rulecount transferred to a 2-fig number in base 36. */
					assign_ruleresp(thisword->resp, rulecount);
					break;
				}
			}
		}

		/* write the current token -- i.e. they only go to prev once they're written */
		if (write_token(thisword, dest) )
		{
			cl_error_file_write(output_filename);
			fcloseall();
			return 1;
		}


		/* scroll 'em down */
		if ( prev[RULESPAN-1] != NULL )
			free(prev[RULESPAN-1]);

		for ( i = (RULESPAN-1) ; i > 0 ; i-- )
			prev[i] = prev[i-1];

		for ( i = 0 ; i < (RULESPAN-1) ; i++ )
			next[i] = next[i+1];

		prev[0] = (thisword = next[0]);
		
		/* get a new 'un in next[RULESPAN-1] ; load_token sets to NULL if end-of-file */
		next[RULESPAN-1] = load_token(source);
	}

	/* free everything in prev */
	for ( i = 0 ; i < RULESPAN ; i++ )
		if (prev[i] != NULL)
			free(prev[i]);


	/* close read and write files */
	if (fclose(source) < 0)
	{
		cl_error_file_close(input_filename);
		fcloseall();
		return 1;
	}
	if (fclose(dest) < 0)
	{
		cl_error_file_close(output_filename);
		fcloseall();
		return 1;
	}

	return 0;
}


void apply_rule(rule  *thisrule,
				token *thisword,
				token *prevword[RULESPAN],
				token *nextword[RULESPAN])
{
	static int times = 0;
	cond *this_cond;

	/* first, check all the conditions; if any is not fulfilled, return */
	/* (if there are no conditions, it will break right away and pass to the action)  */

	for(this_cond = thisrule->condlist ;
		this_cond != NULL ;
		this_cond = this_cond->nextcond )
		{
			if ( ! this_cond->assess(thisword, prevword, nextword, 
				                     this_cond->range, this_cond->matchstring) )
				return;
		}

	/* having got past there, we know that all the conditions have been fulfilled */
	/* therefore, perform the action on the current token */

	thisrule->action(thisword, thisrule->tagstring);
}



void assign_ruleresp(unichar *string, int rulecount)
{
	/* assign first char to R */
	*(string) = 0x0052;

	/* assign second char to number of complete thirty-sixes */
	*(string+1) = rulecount / 36;

	/* assign third char to number of units */
	*(string+2) = rulecount % 36;

	/* second and third char: if they're d0 - d9, add hex 30 */
	/* else if they're d10 to d35, add hex 57 */
	if ( *(string+1) <= 9 )
		*(string+1) += 0x30;
	else if ( *(string+1) >= 10 && *(string+1) <= 35 )
		*(string+1) += 0x57;

	if ( *(string+2) <= 9 )
		*(string+2) += 0x30;
	else if ( *(string+2) >= 10 && *(string+2) <= 35 )
		*(string+2) += 0x57;
}



/* memory allocation functions */

cond *get_cond(void)
{
	cond *pointer;
	if(!(pointer = (cond *)malloc(sizeof(cond))))
	{
		cl_error_memory_alloc();
		fcloseall();
		exit(1);
	}
	/* initialise condition to all-null */
	pointer->assess = NULL;
	pointer->range = 0;
	pointer->nextcond = NULL;

	return pointer;
}

rule *get_rule(void)
{
	rule *pointer;
	if(!(pointer = (rule *)malloc(sizeof(rule))))
	{
		cl_error_memory_alloc();
		fcloseall();
		exit(1);
	}
	/* initialise condition to all-null */
	pointer->action = NULL;
	pointer->condlist = NULL;
	pointer->tagstring[0] = 0;
	pointer->nextrule = NULL;

	return pointer;
}

void free_rulelist(rule *list)
{
	rule *current;
	rule *next;
	cond *c_curr;
	cond *c_next;

	for ( current = list ; current ; current = next )
	{
		next = current->nextrule;

		/* free its condition list */
		for ( c_curr = current->condlist ; c_curr ; c_curr = c_next )
		{
			c_next = c_curr->nextcond;
			free(c_curr);
		}

		free(current);
	}
}




void action_assign(token *word, unichar *matchtag)
{
	clear_tags(word);

	ustrcpy(word->tag[0], matchtag);
}


void action_select(token *word, unichar *matchtag)
{
	int i, j;

	/* if there is no ambiguity, return */
	if ( word->tag[1][0] == 0x0000 )
		return;

	for ( i = 0 ; i < TAGSMAX && word->tag[i][0] ; i++ )
	{
		if ( rustrident(word->tag[i], matchtag ) )
		{
			/* copy this tag to the first tagslot */
			if (i != 0)
			{
				ustrcpy(word->tag[0], word->tag[i]);
				word->prob[0] = word->prob[i];
			}

			/* wipe all the other tags */
			for ( j = 1 ; j < TAGSMAX ; j++ )
			{
				word->tag[j][0] = 0x0000;
				word->prob[j] = 0;
			}
			return;
		}
	}
	/* if the selected tag is not found, no action is taken */
}


void action_delete(token *word, unichar *matchtag)
{
	int i, j;

	for ( i = 0 ; i < TAGSMAX && word->tag[i][0] ; i++ )
	{
		/* if there is no ambiguity, return */
		if ( word->tag[1][0] == 0x0000 )
			return;

		if ( rustrident(word->tag[i], matchtag ) )
		{
			/* overwrite all the later tags down one */
			for ( j = i ; j+1 < TAGSMAX ; j++ )
			{
				if (word->tag[j][0] == 0x0000)
					break;
				ustrcpy(word->tag[j], word->tag[j+1]);
				word->prob[j] = word->prob[j+1];
			}

			/* scroll i back one so it will look again at the one that has just */
			/* been overwritten */
			i--;
		}
	}
}

void action_deletenot(token *word, unichar *matchtag)
{
	int i, j;

	for ( i = 0 ; i < TAGSMAX && word->tag[i][0] ; i++ )
	{
		/* if there is no ambiguity, return */
		if ( word->tag[1][0] == 0x0000 )
			return;

		if ( ! rustrident(word->tag[i], matchtag ) )
		{
			/* overwrite all the later tags down one */
			for ( j = i ; j+1 < TAGSMAX ; j++ )
			{
				if (word->tag[j][0] == 0x0000)
					break;
				ustrcpy(word->tag[j], word->tag[j+1]);
				word->prob[j] = word->prob[j+1];
			}

			/* scroll i back one so it will look again at the one that has just */
			/* been overwritten */
			i--;
		}
	}
}