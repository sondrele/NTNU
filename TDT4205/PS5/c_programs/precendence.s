	.file	"precendence.c"
	.text
	.globl	main
main:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0, %eax
	call	precedence
	movl	$0, %eax
	popq	%rbp
	ret
.LC0:
	.string	"2*(3-1) = %d\n"
.LC1:
	.string	"2*3-1 = %d\n"
	.text
precedence:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	$2, -16(%rbp)
	movl	$3, -12(%rbp)
	movl	$1, -8(%rbp)
	movl	-8(%rbp), %eax
	movl	-12(%rbp), %edx
	movl	%edx, %ecx
	subl	%eax, %ecx
	movl	%ecx, %eax
	imull	-16(%rbp), %eax
	movl	%eax, -4(%rbp)
	movl	-4(%rbp), %eax
	movl	%eax, %esi
	movl	$.LC0, %edi
	movl	$0, %eax
	call	printf
	movl	-16(%rbp), %eax
	imull	-12(%rbp), %eax
	subl	-8(%rbp), %eax
	movl	%eax, -4(%rbp)
	movl	-4(%rbp), %eax
	movl	%eax, %esi
	movl	$.LC1, %edi
	movl	$0, %eax
	call	printf
	movl	$0, %eax
	leave
	ret
