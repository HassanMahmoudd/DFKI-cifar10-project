/*global $, alert, console, jQuery*/

$(document).ready(function () {
    
    'use strict';
    
    $('html').niceScroll({
        
        cursorcolor: '#000',
        curosorwidth: '5px',
        cursorborder: '1px solid #000',
        cursorborderradius: 5
        
    });

    
    //var myHeader = $('.header');
    
    //myHeader.height($(window).height());
    
    $('body, html').width($(window).width());
    
    $(window).resize(function() {
        
       // myHeader.height($(window).height());
        
        $('body, html').width($(window).width());
        
    });
    
    $('.links li a').click(function() {
        
        $(this).parent().addClass('active').siblings().removeClass('active');
        
    });
    
    
    // Cache the window object
    var $window = $(window);
    

    
    $('.shuffle li').click(function() {
        
        $(this).addClass('selected').siblings().removeClass('selected');
        
    });
    
    
    
    var scrollButton = $('#scroll-top');
    
    $(window).scroll(function () {
        
        if ($(this).scrollTop() >= 700) {
            
            scrollButton.show();
            
        } else {
                
            scrollButton.hide();
                
        }
        
        
    });
    
    scrollButton.click(function () {
            
        $('html, body').animate({ scrollTop : 0}, 600);
            
    });
    
});

$(window).load(function () {
    
    
    'use strict';
    
    $('body, html').width($(window).width());
    
//    $('.loading-overlay .spinner').fadeOut(2000, function () {
//        
//        $('body').css("overflow", "auto");
//        
//        $(this).parent().fadeOut(2000, function () {
//            
//            $(this).remove();
//            
//        });
//        
//    });
    
    $('.loading-hm').fadeOut(2000, function () {
        
        $('body').css("overflow", "auto");
            
         $(this).remove();
        
        
    });
    
});