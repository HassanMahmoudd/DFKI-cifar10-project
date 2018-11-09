///*global $, alert, console, jQuery*/
//$(window).load(function () {
//    
//    
//    'use strict';
//    
//    $('body, html').width($(window).width());
//    
//    $('.loading-hm').fadeOut(2000, function () {
//        
//        $('body').css("overflow", "auto");
//        
//        
//            
//        $(this).remove();
//            
//    
//        
//    });
//    
//    
////    $('body').css('visibility', 'hidden');
////    //$('.loading-hm').css('display', 'block');
////    $('.loading-hm').fadeOut(2000, function () {
////        
////        
////        
////        $('body').animate({
////         visibility: "visible"   
////        },1000);
////        
////        $('body').css("overflow", "auto");
////        
////                
////        $(this).remove();
////            
////    
////    });
//    
//});
//$(document).ready(function () {
//    
//    'use strict';
////    $(document).width($(window).width());
//    $('body, html').width($(window).width());
//    
//    $('html').niceScroll({
//        
//        cursorcolor: '#000',
//        curosorwidth: '5px',
//        cursorborder: '1px solid #000',
//        cursorborderradius: 5
//        
//    });
//    
//    if( $(window).width() < 768 ) {
//        console.log($(document).width())
//        console.log($(window).width())
//        
//        
//        $('body, html').width($(window).width());
//        
//        var width = ($(window).width()) + "px";
//
//        var rightTrangleWidth = "0 0 150px " + width;
//
//        var leftTrangleWidth = "150px 0 0 " + width;
//
//        var leftTrangleWidthInstructor = "150px " + width + " 0 0";
//
//        var rightTrangleWidthInstructor = "0 " + width + " 150px 0";
//
//        $('.header .right-triangle').css("border-width", rightTrangleWidth);
//
//        $('.header .left-triangle').css("border-width", leftTrangleWidth);
//
//        $('#instructor .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('#instructor .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('#instructor .right-triangle').css("border-width", rightTrangleWidthInstructor);
//        
//        $('.publications .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('.publications .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('.publications .right-triangle').css("border-width", rightTrangleWidthInstructor);
//        
//        $('.certificates .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('.certificates .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('.certificates .right-triangle').css("border-width", rightTrangleWidthInstructor);
//
//        $('.testim .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('.testim .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('.testim .right-triangle').css("border-width", rightTrangleWidthInstructor);
//
//    }
//    
//    else {
//        $('body, html').width($(window).width());
//        
//        var width = ($(window).width())/2 + "px";
//    
//        var rightTrangleWidth = "0 0 200px " + width;
//
//        var leftTrangleWidth = "200px 0 0 " + width;
//
//        var leftTrangleWidthInstructor = "200px " + width + " 0 0";
//
//        var rightTrangleWidthInstructor = "0 " + width + " 200px 0";
//
//        $('.header .right-triangle').css("border-width", rightTrangleWidth);
//
//        $('.header .left-triangle').css("border-width", leftTrangleWidth);
//
//        $('#instructor .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('#instructor .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('#instructor .right-triangle').css("border-width", rightTrangleWidthInstructor);
//        
//        $('.publications .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('.publications .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('.publications .right-triangle').css("border-width", rightTrangleWidthInstructor);
//        
//        $('.certificates .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('.certificates .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('.certificates .right-triangle').css("border-width", rightTrangleWidthInstructor);
//
//        $('.testim .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//        $('.testim .left-down-triangle').css("border-width", leftTrangleWidth);
//
//        $('.testim .right-triangle').css("border-width", rightTrangleWidthInstructor);
//    
//        
//    }
//    
//    
//    var myHeader = $('.header');
//    
//    myHeader.height($(window).height());
//    
//    $(window).resize(function() {
//        
//        myHeader.height($(window).height());
//        $('body, html').width($(window).width());
//        
//        $('.header .container').css({
//            paddingTop: $('.header').height()/2 - $('.container').height() - 100
//        });
//        
//        if( $(window).width() < 768 ) {
//            $('body, html').width($(window).width());
//            
//            var width = ($(window).width()) + "px";
//    
//            var rightTrangleWidth = "0 0 150px " + width;
//
//            var leftTrangleWidth = "150px 0 0 " + width;
//
//            var leftTrangleWidthInstructor = "150px " + width + " 0 0";
//
//            var rightTrangleWidthInstructor = "0 " + width + " 150px 0";
//            
//            $('.header .right-triangle').css("border-width", rightTrangleWidth);
//
//            $('.header .left-triangle').css("border-width", leftTrangleWidth);
//
//            $('#instructor .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('#instructor .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('#instructor .right-triangle').css("border-width", rightTrangleWidthInstructor);
//            
//            $('.publications .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('.publications .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('.publications .right-triangle').css("border-width", rightTrangleWidthInstructor);
//            
//            $('.certificates .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('.certificates .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('.certificates .right-triangle').css("border-width", rightTrangleWidthInstructor);
//
//            $('.testim .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('.testim .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('.testim .right-triangle').css("border-width", rightTrangleWidthInstructor);
//            
//        }
//        
//        else {
//            $('body, html').width($(window).width());
//            
//            var width = ($(window).width())/2 + "px";
//    
//            var rightTrangleWidth = "0 0 200px " + width;
//
//            var leftTrangleWidth = "200px 0 0 " + width;
//
//            var leftTrangleWidthInstructor = "200px " + width + " 0 0";
//
//            var rightTrangleWidthInstructor = "0 " + width + " 200px 0";
//
//            $('.header .right-triangle').css("border-width", rightTrangleWidth);
//
//            $('.header .left-triangle').css("border-width", leftTrangleWidth);
//
//            $('#instructor .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('#instructor .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('#instructor .right-triangle').css("border-width", rightTrangleWidthInstructor);
//            
//            $('.publications .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('.publications .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('.publications .right-triangle').css("border-width", rightTrangleWidthInstructor);
//            
//            $('.certificates .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('.certificates .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('.certificates .right-triangle').css("border-width", rightTrangleWidthInstructor);
//
//            $('.testim .left-triangle').css("border-width", leftTrangleWidthInstructor);
//
//            $('.testim .left-down-triangle').css("border-width", leftTrangleWidth);
//
//            $('.testim .right-triangle').css("border-width", rightTrangleWidthInstructor);
//            
//        }
//        
//        // Parallax background effect
//        $('div[data-type="background"]').each(function() {
//
//            if($(window).width() > 768) {
//                var $bgobj = $(this); // assigning the object
//
//                $(window).scroll(function() {
//
//                    // scroll the background at var speed
//                    // the yPos is a negative value because we're scrolling it up
//
//                    var yPos = -($window.scrollTop() / $bgobj.data('speed'));
//
//                    // Put together our final background position
//                    var coords = '50% ' + yPos + 'px';
//
//                    // Move the background
//                    $bgobj.css({ backgroundPosition: coords });
//
//                }); // End window scroll
//            }
//
//            else {
//                var $bgobj = $(this); // assigning the object
//
//                var yPos = -(1 / 20);
//
//                // Put together our final background position
//                var coords = '50% ' + yPos + 'px';
//
//                // Move the background
//                $bgobj.css({ backgroundPosition: coords });
//
//                $(window).scroll(function() {
//
//                    // scroll the background at var speed
//                    // the yPos is a negative value because we're scrolling it up
//
//                    var yPos = -(1 / 20);
//
//                    // Put together our final background position
//                    var coords = '50% ' + yPos + 'px';
//
//                    // Move the background
//                    $bgobj.css({ backgroundPosition: coords });
//
//                }); // End window scroll
//            }
//
//        });
//        
//    });
//    
//    $('.header .container').css({
//        paddingTop: $('.header').height()/2 - $('.container').height() - 100
//    });
//    
//    $('.links li a').click(function() {
//        
//        $(this).parent().addClass('active').siblings().removeClass('active');
//        
//    });
//    
//    $('.links li a').click(function() {
//        
//        $('html, body').animate({
//            
//            scrollTop: $('#' + $(this).data('value')).offset().top
//        }, 2000);
//        
//    });
//    
//    $('button.about').click(function () {
//        
//        $('html, body').animate({
//            
//            scrollTop: $('#about').offset().top
//        }, 2000);
//        
//    });
//    
//    $('button.cv').click(function () {
//        
////        window.location.href = "resume.html";
//        window.open(
//          'https://drive.google.com/file/d/15VPHR4UOhv_tadRBK6-SYEk1HRnq4mFl/view?usp=sharing',
//          '_blank' // <- This is what makes it open in a new window.
//        );
//        
//    });
//    
//    $(".features .box i").hover(function(){
//        
//        $(this).siblings().css("color", "white");
//        
//        }, function(){
//        
//        $(this).siblings("h3").css("color", "#505050");
//        
//        $(this).siblings("p").css("color", "#888");
//    });
//    
//    $(".certificates .box i").hover(function(){
//        
//        $(this).siblings().css("color", "white");
//        
//        }, function(){
//        
//        $(this).siblings("h3").css("color", "#505050");
//        
//        $(this).siblings("p").css("color", "#888");
//    });
//    
//    $(".publications .box i").hover(function(){
//        
//        $(this).siblings().css("color", "black");
//        
//        }, function(){
//        
//        $(this).siblings("h3").css("color", "#505050");
//        
//        $(this).siblings("p").css("color", "#888");
//    });
//    
//    // Cache the window object
//    var $window = $(window);
//    
//    // Parallax background effect
//    $('div[data-type="background"]').each(function() {
//        
//        if($(window).width() > 768) {
//            var $bgobj = $(this); // assigning the object
//        
//            $(window).scroll(function() {
//
//                // scroll the background at var speed
//                // the yPos is a negative value because we're scrolling it up
//
//                var yPos = -($window.scrollTop() / $bgobj.data('speed'));
//
//                // Put together our final background position
//                var coords = '50% ' + yPos + 'px';
//
//                // Move the background
//                $bgobj.css({ backgroundPosition: coords });
//
//            }); // End window scroll
//        }
//        
//        else {
//            var $bgobj = $(this); // assigning the object
//            
//            var yPos = -(1 / 20);
//
//            // Put together our final background position
//            var coords = '50% ' + yPos + 'px';
//
//            // Move the background
//            $bgobj.css({ backgroundPosition: coords });
//        
//            $(window).scroll(function() {
//
//                // scroll the background at var speed
//                // the yPos is a negative value because we're scrolling it up
//
//                var yPos = -(1 / 20);
//
//                // Put together our final background position
//                var coords = '50% ' + yPos + 'px';
//
//                // Move the background
//                $bgobj.css({ backgroundPosition: coords });
//
//            }); // End window scroll
//        }
//        
//    });
//    
//    $('#Container').mixItUp({
//        load: {
//		filter: '.branding'
//	}
//    
//});
//    
//    $('.shuffle li').click(function() {
//        
//        $(this).addClass('selected').siblings().removeClass('selected');
//        
////        $('#Container div.mix').each(function () {
////            $(this).css('display', 'none');
////        });
//        
//    });
//    
//    var leftArrow = $('.testim .fa-chevron-left'),
//        
//        rightArrow = $('.testim .fa-chevron-right');
//    
//    function checkClients() {
//        
//        if($('.client:first').hasClass('active')) {
//            
//            leftArrow.hide();
//            
//        }
//        else {
//            leftArrow.fadeIn();
//        }
//        
//        if($('.client:last').hasClass('active')) {
//            
//            rightArrow.hide();
//            
//        }
//        else {
//            rightArrow.fadeIn();
//        }
//        
//    }
//    
////    checkClients();
////    
////    $('.testim i').click(function () {
////        
////        if($(this).hasClass('fa-chevron-right')) {
////
////            $('.testim .active').fadeOut(100, function () {
////                
////                $(this).removeClass('active').addClass('hidden').next('.client').removeClass('hidden').addClass('active').fadeIn(100);
////                
////                checkClients();
////                
////            });
////            
////        }
////        
////        else {
////            
////            $('.testim .active').fadeOut(100, function () {
////                
////                $(this).removeClass('active').addClass('hidden').prev('.client').removeClass('hidden').addClass('active').fadeIn(100);
////                
////                checkClients();
////                
////            });
////        }
////        
////    });
//    
//    (function autoSlider() {
//        
//        $('.testim .active').each(function() {
//            
//            if(!$(this).is(':last-child')) {
//                
//                $(this).delay(7500).fadeOut(1000, function() {
//                    
//                    $(this).removeClass('active').next().addClass('active').fadeIn(1000);
//                    
//                    autoSlider();
//                    
//                });
//                
//            }
//            else {
//                
//                $(this).delay(7500).fadeOut(1000, function() {
//                    
//                    $(this).removeClass('active');
//                    
//                    $('.testim div.client').eq(0).addClass('active').fadeIn(1000);
//                    
//                    autoSlider();
//                    
//                });
//                
//            }
//            
//        });
//        
//    }());
//    
//    var scrollButton = $('#scroll-top');
//    
//    $(window).scroll(function () {
//        
//        if ($(this).scrollTop() >= 700) {
//            
//            scrollButton.show();
//            
//        } else {
//                
//            scrollButton.hide();
//                
//        }
//        
////        if ($(this).scrollTop() < $('#about').offset().top) {
////            
////            $('#instructor').css('background', "url('../images/instructorBG01.png') 100% 50px no-repeat fixed");
////            
////        } else {
////                
////            $('#instructor').css('background', "url('../images/instructorBG01.png') 100% 50px no-repeat");
////                
////        }
//        
//        
//    });
//    
//    scrollButton.click(function () {
//            
//        $('html, body').animate({ scrollTop : 0}, 600);
//            
//    });
//    
//    // Get the modal
//    var modal = document.getElementById('myModal');
//
//    // Get the image and insert it inside the modal - use its "alt" text as a caption
//    //var img = document.getElementById('myImg');
//    var modalImg = document.getElementById("img01");
//    var captionText = document.getElementById("caption");
//    $('.show-img').each(function() {
//        
//        $(this).click(function() {
//            //console.log('hello');
//            modal.style.display = "block";
//            modalImg.src = $(this).siblings('img').data('cover');
//            captionText.innerHTML = $(this).siblings('img').attr('alt');    
//        });
//        
//    });
//
//    modal.onclick = function() {
//        modal.style.display = "none";
//    }
////    img.onclick = function(){
////        modal.style.display = "block";
////        modalImg.src = this.src;
////        captionText.innerHTML = this.alt;
////    }
//    // Get the <span> element that closes the modal
//    var span = document.getElementsByClassName("close")[0];
//
//    // When the user clicks on <span> (x), close the modal
//    span.onclick = function() { 
//      modal.style.display = "none";
//    }
//    
//});
//
