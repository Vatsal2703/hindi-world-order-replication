#include <stdio.h>
#include <stdlib.h>
#include "andrew.h"
#include "unicode.h"
#include "tagfile.h"



/* load a token plus its tags from file; file pointer must be AT START of line,    */
/* and this function cycles to the start of the next line                          */
/* returns a pointer to the word, or NULL for read error (end of file)             */
token *load_token(FILE *source)
{
	int x, y;
	char thistagprobbed, done;

	token *buffer;

	unichar junk;
	unichar s_junk[500];


	buffer = get_token();

	/* get next char : if it's not s, skip line, and try again */
	while (1)
	{
		if ( (junk = fgetuc(source)) == UERR)
		{
			free(buffer);
			return NULL;
		}
		else if (junk == 0x0073)
			break;
		else if ( !fgetus(s_junk, source) )
		{
			free(buffer);
			return NULL;
		}
	}
	/* load the S-idno */
	for (x = 0; x < IDNOLENGTH; x++)
	{
		if((buffer->s_idno[x] = fgetuc(source)) == UERR)
		{
			free(buffer);
			return NULL;
		}
		if(buffer->s_idno[x] == 0x0020)
		{
			buffer->s_idno[x] = 0;
			break;
		}
	}
	/* and null-terminate */
	buffer->s_idno[IDNOLENGTH-1] = 0x0000;


	/* get next char : if it's not w, return NULL (this will end-of file the calling */
	/* function, effectively stopping the prog going any further with a bad file!    */

	if ( (junk = fgetuc(source)) != 0x0077)
	{
		free(buffer);
		return NULL;
	}
	/* load the W-idno */
	for (x = 0; x < IDNOLENGTH; x++)
	{
		if((buffer->w_idno[x] = fgetuc(source)) == UERR)
		{
			free(buffer);
			return NULL;
		}
		if(buffer->w_idno[x] == 0x0020)
		{
			buffer->w_idno[x] = 0;
			break;
		}
	}
	/* and null-terminate */
	buffer->s_idno[IDNOLENGTH-1] = 0x0000;




	/* load the wordform (spotted by the tab) */
	for (x = 0; x < WORDLENGTH; x++)
	{
		if((buffer->wordform[x] = fgetuc(source)) == UERR)
		{
			buffer->wordform[x] = 0;
			break;
		}
		if(buffer->wordform[x] == 0x0009)
		{
			buffer->wordform[x] = 0;
			break;
		}
	}
	/* ensure the string is null-terminated */
	buffer->wordform[WORDLENGTH-1] = 0x0000;


	/* load the resp tag (spotted by space) */
	for (x = 0; x < USERLENGTH; x++)
	{
		if((buffer->resp[x] = fgetuc(source)) == UERR)
		{
			buffer->wordform[x] = 0;
			break;
		}
		if(buffer->resp[x] == 0x0020)
		{
			buffer->resp[x] = 0;
			break;
		}
	}
	/* and null-terminate */
	buffer->resp[USERLENGTH-1] = 0x0000;




	/* load the tags, spotting the probs, if there are any, and loading them */
	done = NO;

	for (y = 0; y < TAGSMAX && done == NO ; y++)
	{
		thistagprobbed = NO;

		for (x = 0; x < TAGLENGTH ; x++)
		{
			/* if the end of the file is reached */
			if ((buffer->tag[y][x] = fgetuc(source)) == UERR)
			{
				done = YES;
				buffer->tag[y][x] = 0;
				break;
			}
			/* assorted checks on things that could signal end of a tag */
			if (buffer->tag[y][x] == 0x0020)
			{
				buffer->tag[y][x] = 0;
				break;
			}
			if (buffer->tag[y][x] == 0x000d)
			{
				done = YES;
				buffer->tag[y][x] = 0;
				break;
			}
			if (buffer->tag[y][x] == 0x002f)
			{
				thistagprobbed = YES;
				buffer->tag[y][x] = 0;
				break;
			}
		}
		buffer->tag[y][WORDLENGTH-1] = 0;

		if (thistagprobbed)
		{
			/* load two chars, change characters to digits, and create frequency score */
			if ((s_junk[0] = fgetuc(source)) == UERR)
				break;
			if ((s_junk[1] = fgetuc(source)) == UERR)
				break;

			s_junk[0] -= 0x0030;
			s_junk[1] -= 0x0030;

			buffer->prob[y] = (10 * s_junk[0]) + s_junk[1];
			s_junk[0] = 0x0000;

			/* load another char. If 000d, then move on out. */
			if ((junk = fgetuc(source)) == UERR)
				break;
			if (junk == 0x000d)
				break;
		}
	}

	/* get to start of next line) */
	while (junk != 0x000a)
		if ((junk = fgetuc(source)) == UERR)
		{
			done = YES;
			break;
		}
	junk = 0;

	return buffer;
}




/* write a token, plus its tags, to file  */
/* returns 0 for all OK, 1 for write error */
int write_token(token *writeme, FILE *dest)
{
	int i, j;

	char c_buffer[5];
	unichar u_buffer[5];
	unichar line[500];

	unichar ess[]   = { 0x0073, 0x0000 };
	unichar dubya[] = { 0x0077, 0x0000 };
	unichar space[] = { 0x0020, 0x0000 };
	unichar tab[]   = { 0x0009, 0x0000 };
	unichar cret[]  = { 0x000d, 0x000a, 0x0000 };
	
	/* assemble a line from the structure */

	ustrcpy(line, ess);

	ustrcat(line, writeme->s_idno);

	ustrcat(line, space);
	ustrcat(line, dubya);

	ustrcat(line, writeme->w_idno);

	ustrcat(line, space);

	ustrcat(line, writeme->wordform);

	ustrcat(line, tab);

	ustrcat(line, writeme->resp);

	ustrcat(line, space);

	/* copy tags. if probs are not zero, then they are written */
	for ( i = 0 ; i < TAGSMAX && writeme->tag[i][0] ; i++ )
	{
		if ( i > 0 )
			ustrcat(line, space);

		ustrcat(line, writeme->tag[i]);
		
		if (writeme->prob[i])
		{
			sprintf(c_buffer, "/%d", writeme->prob[i]);
			for ( j = 0 ; 1 ; j++ )
				if ( ! (u_buffer[j] = (unichar)c_buffer[j]) )
					break;
			ustrcat(line, u_buffer);
		}
	}

	ustrcat(line, cret);


	/* write line to file */
	if (fputus(line, dest) == EOF)
	{
		puts("Error writing token/tag line to file!");
		fcloseall();
		return 1;
	}
	line[0] = 0x0000;

	return 0;
}




/* "get" (i.e. malloc) a structure in which to store a token and its tags */
/* returns pointer to structure, or exits programme for malloc error      */

token *get_token(void)
{
	unsigned char x;
	token *pointer;
	if(!(pointer = (token *)malloc(sizeof(token))))
	{
		error_memory_alloc();
		fcloseall();
		exit(1);
	}
	for (x = 0 ; x < TAGSMAX ; x++)
	{
		pointer->tag[x][0] = 0;
		pointer->prob[x]   = 0;
	}
	pointer->wordform[0] = 0;
	pointer->s_idno[0]   = 0;
	pointer->w_idno[0]   = 0;
	pointer->resp[0]     = 0;

	return pointer;
}



token *copy_token(token *src)
{
	unsigned char x;
	token *pointer;
	if(!(pointer = (token *)malloc(sizeof(token))))
	{
		error_memory_alloc();
		fcloseall();
		exit(1);
	}
	for (x = 0 ; x < TAGSMAX ; x++)
	{
		ustrcpy(pointer->tag[x], src->tag[x]);
		pointer->prob[x] = src->prob[x];;
	}
	ustrcpy(pointer->wordform, src->wordform);
	ustrcpy(pointer->s_idno, src->s_idno);
	ustrcpy(pointer->w_idno, src->w_idno);
	ustrcpy(pointer->resp, src->resp);

	return pointer;
}


void clear_tags(token *word)
{
	int i;

	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		word->tag[i][0] = 0;
		word->prob[i]   = 0;
	}
}



int tagsinclude(token *word, unichar *searchtag)
{
	int i;

	for ( i = 0 ; i < TAGSMAX ; i++)
	{
		if (ustrident(searchtag, word->tag[i]) )
			return TRUE;
	}

	return FALSE;
}
