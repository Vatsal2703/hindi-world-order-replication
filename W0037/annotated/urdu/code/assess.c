#include <stdio.h>
#include <stdlib.h>
#include "andrew.h"
#include "unicode.h"
#include "tagfile.h"
#include "rule.h"
#include "assess.h"

int rustrident(unichar *string1, unichar *string2)
{
	size_t length, length2;
	size_t x;

	length  = ustrlen(string1);
	length2 = ustrlen(string2);

	if (length2 > length)
		length = length2;

	for (x = 0; x <= length; x++)
	{
		/* does either string have the "any number of characters more" character at this point? */
		if ( *(string1+x) == 0x0023 || *(string2+x) == 0x0023 )
			return YES;

		/* does either string have the wildcard character at this point ? */
		/* the wildcard DOES NOT match with the string-terminating NULL   */
		if	(  ( *(string1+x) == 0x002a && *(string2+x) != 0x0000 )
			|| ( *(string2+x) == 0x002a && *(string1+x) != 0x0000 )
			)
			continue;

		/* neither wildcard found, therefore evaluate just on whether the chars match */
		if (*(string1+x) != *(string2+x))
			return NO;
	}
	return YES;
}





/* the assess functions  --  all return YES for fulfilled, NO for not fulfilled */

int assess_ifthiswordis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	/* if identical word token strings between thisword->wordform and matchstring, return YES */
	if (rustrident(thisword->wordform, matchstring))
		return YES;
	else
		return NO;
}

int assess_ifthistagis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	int i;

	/* if any of the tags do not match, return no; otherwise, return yes */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( thisword->tag[i][0] == 0x0000 )
			break;
		if ( ! rustrident(thisword->tag[i], matchstring) )
			return NO;
	}
	return YES;
}

int assess_ifthistaginc(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	int i;

	/* as soon as any tag matches, return yes; otherwise, return no */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( thisword->tag[i][0] == 0x0000 )
			break;
		if ( rustrident(thisword->tag[i], matchstring) )
			return YES;
	}
	return NO;
}

int assess_ifprevwordis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };

	/* if this word is not there (start of file), then return no */
	if ( prevword[r] == NULL )
		return NO;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(prevword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN || prevword[r] == NULL)
			return NO;
	}

	/* if identical strings between prevwords[r]->wordform and matchstring, return YES */
	if ( rustrident(prevword[r]->wordform, matchstring) )
		return YES;
	else
		return NO;
}

int assess_ifprevtagis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (start of file), then return no */
	if ( prevword[r] == NULL )
		return NO;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(prevword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN || prevword[r] == NULL)
			return NO;
	}

	/* if any of the tags do not match, return no; otherwise, return yes */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( prevword[r]->tag[i][0] == 0x0000 )
			break;
		if ( ! rustrident(prevword[r]->tag[i], matchstring) )
			return NO;
	}
	return YES;
}

int assess_ifprevtaginc(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (start of file), then return no */
	if ( prevword[r] == NULL )
		return NO;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(prevword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN || prevword[r] == NULL)
			return NO;
	}

	/* as soon as any tag matches, return yes; otherwise, return no */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( prevword[r]->tag[i][0] == 0x0000 )
			break;
		if ( rustrident(prevword[r]->tag[i], matchstring) )
			return YES;
	}
	return NO;
}

int assess_ifnextwordis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };

	/* if this word is not there (end of file), then return no */
	if ( nextword[r] == NULL )
		return NO;

	/* if nextword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(nextword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN || nextword[r] == NULL)
			return NO;
	}

	/* if identical strings between nextwords[r]->wordform and matchstring, return YES */
	if ( rustrident(nextword[r]->wordform, matchstring) )
		return YES;
	else
		return NO;
}

int assess_ifnexttagis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (end of file), then return no */
	if ( nextword[r] == NULL )
		return NO;

	/* if nextword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(nextword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN || nextword[r] == NULL)
			return NO;
	}

	/* if any of the tags do not match, return no; otherwise, return yes */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( nextword[r]->tag[i][0] == 0x0000 )
			break;
		if ( ! rustrident(nextword[r]->tag[i], matchstring) )
			return NO;
	}
	return YES;
}

int assess_ifnexttaginc(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (end of file), then return no */
	if ( nextword[r] == NULL )
		return NO;

	/* if nextword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(nextword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN || nextword[r] == NULL)
			return NO;
	}

	/* as soon as any tag matches, return yes; otherwise, return no */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( nextword[r]->tag[i][0] == 0x0000 )
			break;
		if ( rustrident(nextword[r]->tag[i], matchstring) )
			return YES;
	}
	return NO;
}


/* NEGATIVE  assess functions -- all return YES for fulfilled, NO for not fulfilled */

int assess_ifthiswordisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	/* if identical word token strings between thisword->wordform and matchstring, return NO */
	if ( rustrident(thisword->wordform, matchstring) )
		return NO;
	else
		return YES;
}


int assess_ifthistagisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	int i;

	/* if any tag does not match, return yes; otherwise return no */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( thisword->tag[i][0] == 0x0000 )
			break;
		if ( ! rustrident(thisword->tag[i], matchstring) )
			return YES;
	}
	return NO;
}

int assess_ifthistagincnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	int i;

	/* as soon as any tag matches, return no; otherwise, return yes */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( thisword->tag[i][0] == 0x0000 )
			break;
		if ( rustrident(thisword->tag[i], matchstring) )
			return NO;
	}
	return YES;
}

int assess_ifprevwordisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };

	/* if this word is not there (start of file), then return yes */
	if ( prevword[r] == NULL )
		return YES;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(prevword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN)
			return NO;
		if (prevword[r] == NULL)
			return YES;
	}

	/* if identical strings between prevwords[r]->wordform and matchstring, return YES */
	if ( rustrident(prevword[r]->wordform, matchstring) )
		return NO;
	else
		return YES;
}

int assess_ifprevtagisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (start of file), then return yes */
	if ( prevword[r] == NULL )
		return YES;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(prevword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN)
			return NO;
		if (prevword[r] == NULL)
			return YES;
	}

	/* if any of the tags do not match, return yes; otherwise, return no */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( prevword[r]->tag[i][0] == 0x0000 )
			break;
		if ( ! rustrident(prevword[r]->tag[i], matchstring) )
			return YES;
	}
	return NO;
}

int assess_ifprevtagincnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (start of file), then return yes */
	if ( prevword[r] == NULL )
		return YES;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(prevword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN)
			return NO;
		if (prevword[r] == NULL)
			return YES;
	}

	/* as soon as any tag matches, return no; otherwise, return yes */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( prevword[r]->tag[i][0] == 0x0000 )
			break;
		if ( rustrident(prevword[r]->tag[i], matchstring) )
			return NO;
	}
	return YES;
}

int assess_ifnextwordisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };

	/* if this word is not there (end of file), then return yes */
	if ( nextword[r] == NULL )
		return YES;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(nextword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN)
			return NO;
		if (nextword[r] == NULL)
			return YES;
	}

	/* if identical strings between nextwords[r]->wordform and matchstring, return NO */
	if ( rustrident(nextword[r]->wordform, matchstring) )
		return NO;
	else
		return YES;
}

int assess_ifnexttagisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (end of file), then return yes */
	if ( nextword[r] == NULL )
		return YES;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(nextword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN)
			return NO;
		if (nextword[r] == NULL)
			return YES;
	}

	/* if any of the tags do not match, return YES; otherwise, return NO */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( nextword[r]->tag[i][0] == 0x0000 )
			break;
		if ( ! rustrident(nextword[r]->tag[i], matchstring) )
			return YES;
	}
	return NO;
}

int assess_ifnexttagincnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring)
{
	const static unichar uniNULL[] = { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };
	int i;

	/* if this word is not there (end of file), then return yes */
	if ( nextword[r] == NULL )
		return YES;

	/* if prevword[r] has a null tag, increase r by one       */
	/* if r = RULESPAN return NO ie the cond is not fulfilled */
	while (ustrident(nextword[r]->tag[0], uniNULL))
	{
		r++;
		if (r == RULESPAN)
			return NO;
		if (nextword[r] == NULL)
			return YES;
	}

	/* as soon as any tag matches, return no; otherwise, return yes */
	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( nextword[r]->tag[i][0] == 0x0000 )
			break;
		if ( rustrident(nextword[r]->tag[i], matchstring) )
			return NO;
	}
	return YES;
}