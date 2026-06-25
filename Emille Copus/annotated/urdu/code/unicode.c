#include <stdio.h>
#include <stdlib.h>
#include "unicode.h"

/* all functions require "unicode.h" - additional header requirements noted below	*/
/* unicode strings to be null-terminated in C										*/

/* gets a binary 1-byte character from an input stream that is open for binary read */
char getbinchar(FILE *source)
{
	char c;
	if (fread(&c, sizeof(char), 1, source) < 1)
		c = EOF;
	return c;
}

/* following ten functions require <stdio.h> */
unichar fputuc(unichar ch, FILE *dest)
{
	/* note: dest must be open for binary write; UERR returned for error			*/
	if (fwrite(&ch, sizeof(unichar), 1, dest) != 1)
		return UERR;
	else
		return ch;
}


char fputus(const unichar *string, FILE *dest)
{
	/* note: dest must be open for binary write; EOF returned for error				*/
	/* alt (older) version:
	unsigned int x;
	for (x = 0 ; 1 ; x++)
	{
		if (!(*(string+x)))
			return 1;
		if (fwrite(string+x, sizeof(unichar), 1, dest) != 1)
			return EOF;
	}
	end alt version */
	unsigned int x;
	x = ustrlen(string);
	if (fwrite(string, sizeof(unichar), x, dest) != x)
		return EOF;
	else
		return 1;
}


unichar fgetuc(FILE *source)
{
	/* note: dest must be open for binary read; UERR returned for error				*/
	unichar ch;
	if (fread(&ch, sizeof(unichar), 1, source) < 1)
		return UERR;
	else
		return ch;
}


int ungetuc(FILE *fp)
{
	/* returns 0 for success, 1 for failure											*/
	/* simulates ungetting using the fseek() function								*/
	/* is not direct analogue: character argument not needed & different returns	*/
	if (fseek(fp, -2, SEEK_CUR))
		return 1;
	else
		return 0;
}


unichar *fgetus(unichar *string, FILE *source)
{
	/* note: NULL returned for error; newline taken to be 000D 000A					*/
	unichar input1 = 0, input2;
	unsigned int x;
	for (x = 0 ; 1 ; x++)
	{
		if (fread(&input2, sizeof(unichar), 1, source) < 1)
			return NULL;
		if (input1 == 0x0d && input2 == 0x0a)
		{
			*(string+x-1) = 0;
			break;
		}
		if (x > 0)
			*(string+x-1) = input1;
		input1 = input2;
	}
	return string;
}


unichar *fgetnus(unichar *string, size_t n, FILE *source)
{
	/* note: NULL returned for error; no newline interpretation; NULL is added		*/
	unsigned int x;
	for (x = 0 ; x < n ; x++)
	{
		if (fread(string+x, sizeof(unichar), 1, source) < 1)
			return NULL;
	}
	*(string+x) = 0;
	return string;
}


size_t ustrlen(const unichar *string)
{
	size_t x = 0;
	while (*(string+x++));
	return x-1;		/* returns no. unicharacters, excluding terminating NULL		*/
}


unichar *ustrcpy(unichar *dest, const unichar *source)
{
	unsigned int x = 0;
	while (*(dest+x) = *(source+x))
		x++;
	return dest;
}


unichar *ustrncpy(unichar *dest, const unichar *source, size_t n)
{
	char overrun = 0;
	unsigned int x = 0;
	while (x < n)
	{
		if (overrun == 0)
			*(dest+x) = *(source+x);
		else
			*(dest+x) = 0;
		if (!(*(source+x++)))
			overrun = 1;
	}
	return dest;
}


/* following function requires <stdlib.h> */
unichar *ustrdup(const unichar *source)
{
	unichar *dest;
	if (!(dest = malloc((ustrlen(source)+1) * sizeof(unichar))))
		return NULL;
	else
		return (ustrcpy(dest, source));
}


unichar *ustrcat(unichar *dest, const unichar *source)
{
	unsigned int x=0, y=0;
	while(*(dest+x))
		x++;
	do
		*(dest+x++) = *(source+y);
	while (*(source+y++));
	return dest;
}


unichar *ustrncat(unichar *dest, const unichar *source, size_t n)
{
	unsigned int x = 0, y= 0;
	while(*(dest+x))
		x++;
	while (*(source+y) && y < n)
		*(dest+x++) = *(source+y++);
	*(dest+x++) = 0, y++;
	while (y++ < n)
		*(dest+x++) = 0;
	return dest;
}


/* note that unicode values can really tot up... may not give accurate answer for high codes/long strings!	*/
long ustrcmp(unichar *str1, unichar *str2)
{
	long total1 = 0, total2 = 0;
	unsigned short x = 0;
	while (*(str1+x))
		total1 += *(str1+x++);
	x = 0;
	while (*(str2+x))
		total2 += *(str2+x++);
	return (total1 - total2);
}


unichar *ustrchr(unichar *string, const unichar chr)
{
	while (*string)
	{
		if ((*string) == chr)
			return string;
		string++;
	}
	return NULL;
}




unichar *ustrrchr(unichar *str, const unichar chr)
{
	unichar *string = str;
	while(*string)
		string++;
	while(string != str)
	{
		if ((*string) == chr)
			return string;
		string--;
	}
	return NULL;
}


size_t ustrcspn(unichar *str1, unichar *str2)
{
	size_t x, y;
	for(x = 0; *(str1+x); x++)
		for (y = 0; *(str2+y); y++)
			if (*(str1+x) == *(str2+y))
				return x;
	return x;
}


size_t ustrspn(unichar *str1, unichar *str2)
{
	size_t x, y;
	for(x = 0; *(str1+x); x++)
	{
		for (y = 0; *(str2+y); y++)
			if (*(str1+x) == *(str2+y))
				break;
		if (*(str1+x) == *(str2+y))
			continue;
		return x;
	}			
	return x;
}


/* next needs <stdlib.h>; return NULL in case of malloc() error.					*/
unichar *ustrrev(unichar *string)
{
	size_t x;
	unichar *basis;
	unichar *temp;
	basis = string + ((x = ustrlen(string)) - 1);
	if (!(temp = malloc((x+1) * sizeof(unichar))))
		return NULL;
	for (x = 0; basis-x >= string; x++)
		*(temp+x) = *(basis-x);
	*(temp+x) = 0;
	ustrcpy(string, temp);
	free(temp);
	return string;
}


/* next needs <stdlib.h>; return NULL in case of malloc() error.				*/
unichar *make_unicode(const char *string)
{
	unichar *pntr;
	size_t x = 0;
	while(*(string+x++));
	if (!(pntr = malloc(x * sizeof(unichar))))
		return NULL;
	for (x = 0; *(string+x); x++)
		*(pntr+x) = *(string+x);
	*(pntr+x) = 0;
	return pntr;
}


int ustrident(const unichar *str1, const unichar *str2)
{
	size_t length;
	size_t x;
	if ((length = ustrlen(str1)) != ustrlen(str2))
		return 0;
	for (x = 0; x <= length; x++)
		if (*(str1+x) != *(str2+x))
			return 0;
	return 1;
}


/* next returns number of characters from start for which the two strings match	*/
size_t ustrnident(const unichar *string1, const unichar *string2)
{
	size_t cntr = 0;
	while (*(string1) == *(string2++))
	{
		cntr++;
		if (!(*(string1++)))
			break;
	}
	return cntr;
}


void ustrimstart(unichar *str, int n)
{
	int x = 0;

	while (1)
	{
		*(str+x) = *(str + (x+n) );
		if ( *(str+x) == 0 )
			break;
		x++;
	}

//	while ( (*(str+x) = *(str+x+n)) )
//		x++;
}


void ustrimend(unichar *str, int n)
{
	int x = 0;
	while ( *(str+x) )
		x++;
	*(str + x - n) = 0;
}


/* following two functions require <stdio.h>, return TRUE if directionality correct	*/
int ucheckdir(FILE *file)
{
	if (fgetuc(file) != RIGHTWAY)
		return 0;
	else
		return 1;
}


/* character test functions															*/
int whitespace(unichar ch)
{
	/* note: U+ffff is considered to be whitespace since it is returned by some of  */
	/* the functions above when they encounter end-of-file							*/
	/* returns 1 for a whitespace character, 0 for any other character				*/
	if (ch == 0xffff ||		/* UERR */
		// ?? ch == 0xfeff ||		/* zero-width no-break space (RIGHTWAY) */
		ch == 0x3000 || 	/* ideographic space (Chinese writing system) */
		ch == 0x2029 ||		/* paragraph separator */
		ch == 0x2028 || 	/* line separator */
		ch == 0x200f ||		/* right-to-left mark */
		ch == 0x200e || 	/* left-to-right mark */
		// ?? ch == 0x200d ||		/* zero-width joiner */
		ch == 0x200c ||		/* zero-width non-joiner */
		ch == 0x200b ||		/* zero-width space */
		ch == 0x200a || 	/* hair space */
		ch == 0x2009 ||		/* thin space */
		ch == 0x2008 || 	/* punctuation space */
		ch == 0x2007 ||		/* figure space */
		ch == 0x2006 || 	/* six-per-em space */
		ch == 0x2005 ||		/* four-per-em space */
		ch == 0x2004 || 	/* three-per-em space */
		ch == 0x2003 ||		/* em space */
		ch == 0x2002 || 	/* en space */
		ch == 0x2001 ||		/* em quad */
		ch == 0x2000 || 	/* en quad */
		// ?? ch == 0x00a0 ||		/* no-break space */
		ch == 0x0020 || 	/* space */
		ch == 0x000d ||		/* carriage return */
		ch == 0x000b || 	/* vertical tabulation */
		ch == 0x000a ||		/* line feed */
		ch == 0x0009 )		/* horizontal tabulation */
		return 1;
	else
		return 0;
}

void udecaps(unichar *string)
{
	int i;

	for ( i = 0 ; *(string + i) ; i++ )
		if ( *(string+i) > 0x0040 && *(string+i) < 0x005b )
			*(string+i) += 0x0020;
	/* ideally, there would be Greek and Cyrillic here as well. But not for now. */
}


int punctuation(unichar ch)
/* punctuation is defined as characters not separated from other words by whitespace, */
/* but considered to be separate words for tagging purposes */
/* it does not include mathematical operators, which are not separate words but */
/* do suggest that their word is a formaula */
/* it does not include angled brackets., because these are SGML indicators -- */
/* -- to be dealt with separately */
/* **doesn't** include hyphen since hypen/minus is often word-internal */
/* the break dash is enclosed in whitespace */
/* Does not cover dingbats */
{
	if (ch == 0x0021 || /* exclamation mark */
		ch == 0x0022 || /* neutral quotation mark */
		ch == 0x0026 || /* ampersand */
		ch == 0x0027 || /* neutral apostrophe */
		ch == 0x0028 || /* open parenthesis */
		ch == 0x0029 || /* close parenthesis */
		ch == 0x002a || /* asterisk */
		ch == 0x002c || /* comma */
		/*ch == 0x002d || /* hyphen / minus */
		ch == 0x002e || /* full stop */
		ch == 0x002f || /* forward slash ("solidus") */
		ch == 0x003a || /* colon */
		ch == 0x003b || /* semicolon */
		ch == 0x003f || /* question mark */
		ch == 0x005b || /* open square bracket */
		ch == 0x005d || /* close square bracket */
		ch == 0x007b || /* open brace */
		ch == 0x007d || /* close brace */
		ch == 0x00a1 || /* inverted exclamation mark */
		ch == 0x00a7 || /* paragraph sign */
		ch == 0x00ab || /* left guillemet */
		ch == 0x00b6 || /* pilcrow sign */
		ch == 0x00b7 || /* middle dot */
		ch == 0x00bb || /* right guillemet */
		ch == 0x00bf || /* inverted question mark */
		ch == 0x02bc || /* apostrophe (preferred for elision) */
		ch == 0x037e || /* Greek question mark */
		ch == 0x0387 || /* Greek middle dot (not sure about this!) */
/* not sure about the armenian characters other than full stop:
are they sounds or punctuation? */
		ch == 0x055a || /* Armenian apostrophe */
		ch == 0x055c || /* Armenian exclamation mark */
		ch == 0x055d || /* Armenian comma */
		ch == 0x055e || /* Armenian question mark */
		ch == 0x055f || /* Armenian abbreviation mark */
		ch == 0x0589 || /* Armenian full stop */
		ch == 0x05be || /* Hebrew maqaf */
		ch == 0x05c0 || /* Hebrew pasaq */
		ch == 0x05c3 || /* Hebrew sof pasuq */
		ch == 0x05f3 || /* Hebrew geresh */
		ch == 0x05f4 || /* Hebrew gershayim */
		ch == 0x060c || /* Arabic comma */
		ch == 0x061b || /* Arabic semicolon */
		ch == 0x061f || /* Arabic question mark */
		ch == 0x06d4 || /* Arabic (Urdu) full stop */
		ch == 0x06dd || /* Arabic end of Ayah */
		ch == 0x06de || /* Arabic start of rub el hizb */
		ch == 0x06e9 || /* Arabic place of sajdah */
		ch == 0x0970 || /* Devanagari abbreviation sign */
		ch == 0x10fb || /* Georgian paragraph separator */
		(ch > 0x200f && ch < 0x2016 ) || /* hyphens and dashes */
		(ch > 0x2017 && ch < 0x2028 ) || /* quotation marks, bullets, ellipsis, leader dots */
		(ch > 0x2039 && ch < 0x203e ) || /* quotation marks, reference mark, double exclamation mark, interrobang */
		ch == 0x2042 || /* asterism */
		ch == 0x2043 || /* hyphen bullet */
		ch == 0x2045 || /* open square bracjket with quill */
		ch == 0x2046 || /* close square bracjket with quill */
		ch == 0x207d || /* open superscript parenthesis */
		ch == 0x207e || /* close superscript parenthesis */
		ch == 0x208d || /* open subscript parenthesis */
		ch == 0x208e || /* close subscript parenthesis */
		(ch > 0x3000 && ch < 0x3021 ) || /* CJK punctuation - could probably be refined !*/
		ch == 0x30fb || /* Katakana middle dot */
		ch == 0xfd3e || /* Arabic ornate parenthesis */
		ch == 0xfd3f || /* Arabic ornate right parenthesis */
		(ch > 0xfe29 && ch < 0xfe5f ) || /* CJK compatability punctuation, small forms */ 
		ch == 0xfe60 || /* small ampersand */
		ch == 0xfe61 || /* small asterisk */
		ch == 0xff01 || /* fullwidth exclamation mark */
		ch == 0xff02 || /* fullwidth neutral double quotation mark */
		(ch > 0xff05 && ch < 0xff0b ) || /* fullwidth punctuation */
		ch == 0xff0c || /* fullwidth comma */
		ch == 0xff0e || /* fullwidth full stop */
		ch == 0xff1a || /* fullwidth colon */
		ch == 0xff1b || /* fullwidth semi-colon */
		ch == 0xff1f || /* fullwidth question mark */
		ch == 0xff3b || /* fullwidth open square bracket */
		ch == 0xff3d || /* fullwidth close square bracket */
		ch == 0xff5b || /* fullwidth open brace */
		ch == 0xff5d )  /* fullwidth close brace */
		return 1;
	else
		return 0;
}